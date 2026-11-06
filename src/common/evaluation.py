"""首题与有限恢复分开计量，未作答、异常都保留在各自分母中。"""
from collections import Counter


def summarize_reports(reports):
    attempts = [a for r in reports for a in r['attempts']]
    first = [r['attempts'][0] if r['attempts'] else {'outcome': 'unknown'} for r in reports]
    retries = [a for r in reports for a in r['attempts'][1:]]
    submitted = [a for a in attempts if a.get('receipts')]
    recovered = sum(r['verdict'] == 'passed' and r['attempts'][0]['outcome'] != 'passed' for r in reports)
    first_passed = sum(a['outcome'] == 'passed' for a in first)
    refreshes = [event for r in reports for event in r.get('refreshes', [])]
    return {'cases': len(reports), 'passed': sum(r['verdict'] == 'passed' for r in reports),
            'failed': sum(r['verdict'] == 'failed' for r in reports),
            'unknown': sum(r['verdict'] == 'unknown' for r in reports),
            'first_attempt_passed': first_passed,
            'first_attempt_outcomes': dict(Counter(a['outcome'] for a in first)),
            'first_attempt_usable': sum(a.get('prediction', {}).get('usable') is True for a in first),
            'first_attempt_submitted': sum(bool(a.get('receipts')) for a in first),
            'first_not_passed_cases': len(reports)-first_passed, 'recovered_cases': recovered,
            'retry_attempts': len(retries), 'retry_passed': sum(a['outcome'] == 'passed' for a in retries),
            'retry_outcomes': dict(Counter(a['outcome'] for a in retries)),
            'attempts': len(attempts), 'outcomes': dict(Counter(a['outcome'] for a in attempts)),
            'by_attempt': {str(i+1): dict(Counter(r['attempts'][i]['outcome'] for r in reports if len(r['attempts']) > i))
                           for i in range(max((len(r['attempts']) for r in reports), default=0))},
            'refresh_sources': dict(Counter(e['source'] for e in refreshes)) if all('refreshes' in r for r in reports) else None,
            'server_submissions': len(submitted),
            'accepted_submissions': sum(any(x.get('accepted') is True for x in a['receipts']) for a in submitted),
            'exceptions': sum('error' in r for r in reports),
            'negative_control': any(r.get('negative_control') for r in reports)}
