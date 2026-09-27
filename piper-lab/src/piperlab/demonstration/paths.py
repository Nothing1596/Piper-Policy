"""Rebase nested evidence paths when a demonstration bundle is copied."""
from pathlib import Path


def relocate_bundle(value,old_root,new_root):
    old_root=Path(old_root).absolute();new_root=Path(new_root)
    def visit(item):
        if isinstance(item,list):return [visit(x) for x in item]
        if not isinstance(item,dict):return item
        result={key:visit(val) for key,val in item.items()}
        for key in ('image_path','source_path','path'):
            if not isinstance(result.get(key),str):continue
            path=Path(result[key])
            if path.is_absolute() and path.is_relative_to(old_root):
                result[key]=str(new_root/path.relative_to(old_root))
        return result
    return visit(value)


def resolve_bundle(value,root):
    """Prefer archived evidence next to demo.json, even if its old location exists."""
    root=Path(root).absolute()
    source=Path(value.get('source_path',''))
    if source.name.startswith('source.') and (root/source.name).is_file():
        value=relocate_bundle(value,source.parent,root)
    def visit(item):
        if isinstance(item,list):return [visit(x) for x in item]
        if not isinstance(item,dict):return item
        result={key:visit(val) for key,val in item.items()}
        for key in ('image_path','source_path'):
            if isinstance(result.get(key),str) and not Path(result[key]).is_absolute():
                result[key]=str(root/result[key])
        return result
    return visit(value)
