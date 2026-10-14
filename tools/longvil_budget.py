"""Append-only usage estimates in CNY; one serial request, no credential storage."""
from datetime import datetime,timedelta,timezone
import json
import math
from pathlib import Path
from contextlib import contextmanager

class BudgetPause(RuntimeError):pass


@contextmanager
def ledger_lock(path):
    """One local Mac/Linux job owns the shared spending ledger at a time."""
    import fcntl
    lock=Path(str(path)+'.lock');lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a') as stream:
        try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise BudgetPause('Another offline job owns this budget ledger') from None
        try:yield
        finally:fcntl.flock(stream,fcntl.LOCK_UN)


def peak_overlap(start,end):
    tz=timezone(timedelta(hours=8));a=start.astimezone(tz);b=end.astimezone(tz)
    day=a.replace(hour=0,minute=0,second=0,microsecond=0)
    while day<=b:
        if day.weekday()<5:
            for lo,hi in ((9,12),(14,18)):
                if a < day+timedelta(hours=hi) and b >= day+timedelta(hours=lo):return True
        day+=timedelta(days=1)
    return False


def usage_cost(usage,start,end):
    keys=('prompt_cache_hit_tokens','prompt_cache_miss_tokens','completion_tokens')
    if not isinstance(usage,dict) or any(type(usage.get(k)) is not int or usage[k]<0 for k in keys):
        raise BudgetPause('Usage missing or invalid; reserve 3 CNY and pause')
    factor=2 if peak_overlap(start,end) else 1
    return factor*(usage[keys[0]]*.02+usage[keys[1]]+usage[keys[2]]*4)/1_000_000


class Ledger:
    def __init__(self,path,*,limit=75,stop_at=72):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.limit=None if limit is None else float(limit);self.stop_at=None if stop_at is None else float(stop_at)
        self.rows=[json.loads(line) for line in self.path.read_text().splitlines()] if self.path.exists() else []
        if (self.limit is None)!=(self.stop_at is None):raise ValueError('Both spending thresholds must be set or disabled')
        if self.limit is not None and (not math.isfinite(self.limit) or not 0<self.stop_at<=self.limit-3):raise ValueError('Need 3 CNY reserve')

    @property
    def total(self):return sum(row['estimated_cny'] for row in self.rows)

    def check(self):
        settled={row.get('reconciles_request_id') for row in self.rows
                 if row['status']=='conservative_reserve_accepted' and row['estimated_cny']==0}
        if any(row['status']=='usage_unavailable' and (row['request_id'] not in settled or row['estimated_cny']!=3)
               for row in self.rows):raise BudgetPause('Unreconciled usage; explicit reconciliation required')
        if self.stop_at is not None and self.total>=self.stop_at:raise BudgetPause('Budget checkpoint reached')

    def settle_conservative(self,request_id):
        """Explicit operator settlement; retain the full 3 CNY charge, never invent usage."""
        original=next((r for r in self.rows if r['request_id']==request_id),None)
        if not original or original['status']!='usage_unavailable' or original['estimated_cny']!=3:
            raise ValueError('Only an unknown-usage request with its full 3 CNY reserve can be settled')
        if any(r.get('reconciles_request_id')==request_id for r in self.rows):raise ValueError('Already settled')
        now=datetime.now(timezone.utc).isoformat()
        row={'request_id':'settlement/'+request_id,'phase':'reconciliation','start':now,'end':now,
             'status':'conservative_reserve_accepted','usage':None,'estimated_cny':0.,'peak_rate':False,
             'reconciles_request_id':request_id,'price_snapshot':original['price_snapshot'],'error_type':None,
             'note':'Operator explicitly resumed. Original 3 CNY remains charged as a conservative estimate; actual supplier usage is still unknown.'}
        with self.path.open('a',encoding='utf-8') as stream:stream.write(json.dumps(row)+'\n')
        self.rows.append(row)
        return row

    def record(self,*,request_id,phase,usage,start,end,error_type=None):
        if any(row['request_id']==request_id for row in self.rows):raise ValueError('Duplicate usage request')
        try:cost=usage_cost(usage,start,end);status='accounted'
        except BudgetPause:cost=3.;status='usage_unavailable'
        row={'request_id':request_id,'phase':phase,'start':start.isoformat(),'end':end.isoformat(),
             'status':status,'usage':usage,'estimated_cny':cost,'peak_rate':peak_overlap(start,end),
             'price_snapshot':'2026-10-09 official CNY flash .02/1/4 off-peak, double peak','error_type':error_type}
        with self.path.open('a',encoding='utf-8') as stream:stream.write(json.dumps(row)+'\n')
        self.rows.append(row)
        if status=='usage_unavailable':raise BudgetPause('Usage unavailable; budget reserved and job paused')
        return cost
