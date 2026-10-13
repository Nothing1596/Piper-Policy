"""Strict symbolic action extraction. Parses AST; never executes generated code."""
import ast

RELATIONS={'left','right','front','behind','top','into'}


def semantic_actions(blocks, objects=None):
    if not isinstance(blocks,list) or not all(isinstance(x,str) for x in blocks):raise ValueError('code must be a list of strings')
    text='\n'.join(blocks)
    if len(text)>200000:raise ValueError('program exceeds analysis budget')
    text=text.replace('<begin_code>','').replace('<end_code>','')
    text='\n'.join(line for line in text.splitlines() if not line.strip().startswith('```'))
    nodes=ast.parse(text).body
    variables={};target=None;held=None;actions=[]
    for node in nodes:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            call=node.value
            if not isinstance(call,ast.Call) or not isinstance(call.func,ast.Name) or call.func.id!='getpos' or call.keywords or len(call.args) not in (1,2):raise ValueError('Only name=getpos literals are allowed')
            values=[]
            for arg in call.args:
                if not isinstance(arg,ast.Constant) or not isinstance(arg.value,str):raise ValueError('getpos arguments must be string literals')
                values.append(arg.value)
            if objects is not None and values[0] not in objects:raise ValueError('Unknown object')
            if len(values)==2 and values[1] not in RELATIONS:raise ValueError('Unknown spatial relation')
            variables[node.targets[0].id]=tuple(values)
        elif isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name):
            call=node.value;name=call.func.id
            if call.keywords:raise ValueError('Keyword calls are not primitives')
            if name=='moveto':
                if len(call.args)!=1 or not isinstance(call.args[0],ast.Name) or call.args[0].id not in variables:raise ValueError('moveto needs a defined position variable')
                target=variables[call.args[0].id]
            elif name=='pick':
                if call.args or held is not None or target is None or len(target)!=1:raise ValueError('pick needs an unheld object target')
                held=target[0]
            elif name=='place':
                if call.args or held is None or target is None or len(target)!=2:raise ValueError('place needs a held object and reference relation')
                actions.append(('pick_place',held,target[0],target[1]));held=None
            elif name in ('open','close'):
                if call.args or held is not None:raise ValueError('Drawer primitive has no arguments or held object')
                actions.append((name,))
            else:raise ValueError('Unsupported statement: '+name)
        else:raise ValueError('Unsupported statement type: '+type(node).__name__)
        if len(actions)>128:raise ValueError('Too many semantic actions')
    if held is not None:raise ValueError('Unmatched pick')
    if not actions:raise ValueError('Empty action program')
    return actions


def compare_actions(reference,predicted):
    n,m=len(reference),len(predicted)
    dp=[[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1):dp[i][0]=i
    for j in range(m+1):dp[0][j]=j
    for i in range(1,n+1):
        for j in range(1,m+1):dp[i][j]=min(dp[i-1][j]+1,dp[i][j-1]+1,dp[i-1][j-1]+(reference[i-1]!=predicted[j-1]))
    missing=extra=spatial=other=0;i,j=n,m
    while i or j:
        if i and j and dp[i][j]==dp[i-1][j-1]+(reference[i-1]!=predicted[j-1]):
            a,b=reference[i-1],predicted[j-1]
            if a!=b:
                if len(a)==len(b)==4 and a[:2]==b[:2]:spatial+=1
                else:other+=1
            i-=1;j-=1
        elif i and dp[i][j]==dp[i-1][j]+1:missing+=1;i-=1
        else:extra+=1;j-=1
    return {'semantic_exact':reference==predicted,'semantic_step_score':max(0.,1-dp[n][m]/max(n,m,1)),
            'missing_actions':missing,'extra_actions':extra,'spatial_errors':spatial,'other_substitutions':other,
            'order_mismatch':sorted(reference)==sorted(predicted) and reference!=predicted,
            'reference_actions':reference,'predicted_actions':predicted}
