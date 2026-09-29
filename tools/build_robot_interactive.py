"""Build a robot-only source + wheel kit; no credentials, runtime DBs or SDK blobs.

Pass an already-built wheel. Installation downloads platform dependencies;
this kit is not described as an offline or physical-hardware-validated bundle.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--wheel',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1]
    version=args.wheel.name.split('-')[1]
    name=f'piper-robot-{version}-interactive'
    root=args.output/name
    root.mkdir(parents=True,exist_ok=False)
    shutil.copy2(args.wheel,root/args.wheel.name)
    source=root/'source'/'piperx-cli'
    source.parent.mkdir()
    shutil.copytree(repo/'piperx-cli',source,ignore=shutil.ignore_patterns(
        '__pycache__','.pytest_cache','*.egg-info','build','dist','.venv*','*.token','*.sqlite3','*.log'))
    shutil.copytree(repo/'docs'/'implementation',root/'implementation')
    shutil.copy2(repo/'docs'/'ROBOT-INTERACTIVE.zh.md',root/'README.zh.md')
    shutil.copy2(repo/'docs'/'ONE-TASK-DEMO.zh.md',root/'ONE-TASK-DEMO.zh.md')
    shutil.copy2(repo/'gpt.md',root/'source'/'gpt.md')
    if (repo/'LICENSE').exists():shutil.copy2(repo/'LICENSE',root/'LICENSE')
    # An isolated venv; never edits an existing installation or robot profile.
    (root/'install.py').write_text('''"""Install into this kit's .venv; downloads platform dependencies."""
import pathlib,subprocess,sys,venv
r=pathlib.Path(__file__).resolve().parent
if sys.version_info<(3,11):raise SystemExit('Python 3.11+ required; Windows CANDO requires x64 Python.')
p=r/'.venv'
if p.exists():raise SystemExit('Existing .venv preserved. Use a fresh extracted kit for an upgrade.')
venv.EnvBuilder(with_pip=True).create(p)
exe=p/('Scripts/python.exe' if sys.platform=='win32' else 'bin/python')
wheels=list(r.glob('piperx_middleware-*.whl'))
if len(wheels)!=1:raise SystemExit('Expected exactly one project wheel')
subprocess.run([str(exe),'-m','pip','install',str(wheels[0])+'[simulation,hardware]'],check=True)
subprocess.run([str(exe),'-m','pip','check'],check=True)
subprocess.run([str(exe),'-m','piperx_middleware.standalone_cli','--help'],check=True)
print('Installed. Launch piper-robot.cmd (Windows) or ./piper-robot (macOS/Linux). No hardware opened.')
''',encoding='utf-8')
    (root/'Setup.cmd').write_text('@echo off\r\npy -3 "%~dp0install.py"\r\nexit /b %errorlevel%\r\n')
    (root/'piper-robot.cmd').write_text('@echo off\r\n"%~dp0.venv\\Scripts\\python.exe" -m piperx_middleware.standalone_cli %*\r\nexit /b %errorlevel%\r\n')
    (root/'piper-robot').write_text('#!/bin/sh\nTASK_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)\nexec "$TASK_DIR/.venv/bin/python" -m piperx_middleware.standalone_cli "$@"\n')
    (root/'piper-robot').chmod(0o755)
    (root/'INSTALL.txt').write_text('''安装：需要 Python 3.11+；Windows CANDO 使用 x64 Python（WOA 的 x64 模拟）。
首次安装下载依赖，需要网络。这不是离线包，不包含厂商 CAN 驱动或 SDK。
Windows：在包目录运行 py -3 install.py，随后 piper-robot.cmd。
macOS/Linux：python3 install.py，随后 ./piper-robot。
也可在自己的虚拟环境 pip install ./piperx_middleware-*.whl 后运行 piper-robot；仿真另装 [simulation] extra。
安装和 --help 不打开 CAN。请先选择仿真验证。真机连接仍需厂商驱动、SDK 和真实反馈。
升级：退出旧前端/执行器，解压到新目录，不覆盖正在运行的旧包或配置。
用户配置不在发行包中；首次运行按独立模式/目标迁移，保留限制。
完整操作说明 README.zh.md；单次任务演示 ONE-TASK-DEMO.zh.md；验证范围 implementation/validation.md。
''',encoding='utf-8')
    manifest=[{'path':p.relative_to(root).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
              for p in sorted(root.rglob('*')) if p.is_file()]
    (root/'SHA256.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    archive=args.output/(name+'.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(root.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(args.output))
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix('.zip.sha256').write_text(digest+'  '+archive.name+'\n')
    print(json.dumps({'archive':str(archive),'sha256':digest,'files':len(manifest),'bytes':archive.stat().st_size}))

if __name__=='__main__':main()
