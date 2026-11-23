"""Evidence-grounded feature extraction, optional hosted LLM interface.

Offline mode is a deterministic evidence-spanned numeric parser on synthetic text.
It is a proxy, not an LLM. Anthropic API is optional and must be explicitly opted into.
"""
import json, os, re, time
import numpy as np
from .data import STATE_NAMES

FIELDS = [('interview','belief'), ('observation','practice'),('pd_record','context'),('reflection_text','reflection')]


def encode_offline(row):
    scores, spans, conf = [], [], []
    for evidence_field, construct in FIELDS:
        evidence = str(row[evidence_field])
        # Labels printed in the generated vignettes create a transparent oracle-like
        # diagnostic; it is NOT equivalent to classification of authentic narratives.
        patterns = {
            'belief': r'reported belief (\d+\.\d+)',
            'practice': r'practice at (\d+\.\d+)',
            'context': r'contextual support (\d+\.\d+)',
            'reflection': r'reflection score (\d+\.\d+)'
        }
        match = re.search(patterns[construct], evidence)
        scores.append(float(match.group(1)) if match else 3.)
        spans.append(match.group(0) if match else '')
        conf.append(.95 if match else .15)
    return np.asarray(scores, dtype=np.float32), spans, np.asarray(conf, dtype=np.float32)


def encode_anthropic(row, model='claude-sonnet-4-6'):
    if not os.getenv('ANTHROPIC_API_KEY'):
        raise RuntimeError('Set ANTHROPIC_API_KEY to use --encoder anthropic')
    try:
        import anthropic
    except ImportError as exc:
        raise ImportError('pip install anthropic') from exc
    client=anthropic.Anthropic()
    texts={k:str(row[k]) for k,_ in FIELDS}
    msg=client.messages.create(model=model,max_tokens=450,temperature=0,
        system='You are a strict evidence coder. Return only a JSON object with keys belief, practice, context, reflection. Each key must have score (float 1..5), span (exact quote from supplied text), confidence (float 0..1). Never infer unsupported scores; use score 3 and confidence 0 when evidence is missing.',
        messages=[{'role':'user','content':json.dumps(texts)}])
    raw=''.join(part.text for part in msg.content if getattr(part,'type','')=='text')
    data=json.loads(raw.strip().removeprefix('```json').removesuffix('```').strip())
    values=[]; spans=[]; conf=[]
    for field,construct in FIELDS:
        item=data[construct]; span=str(item.get('span',''))
        verified=bool(span) and span in texts[field]
        values.append(float(np.clip(item.get('score',3),1,5)))
        spans.append(span if verified else '')
        conf.append(float(np.clip(item.get('confidence',0),0,1)) if verified else 0.)
    return np.array(values,np.float32),spans,np.array(conf,np.float32)
