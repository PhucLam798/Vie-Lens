"""Distinct synthetic source-to-display checks for every result-table builder."""
from __future__ import annotations

import csv
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from scripts import tables


def synthetic_sources(repo: Path) -> None:
    """Create tiny independent source values across every family/cell/variant."""
    records: dict[str,list[dict[str,str]]] = {name:[] for name in tables.KEYS}
    counter=0
    def value() -> str:
        nonlocal counter
        counter+=1
        return str(Decimal(counter)/Decimal('137')+Decimal('.123456789'))
    def add(name: str, **data: Any) -> None:
        row={key:'' for key in tables.KEYS[name]}
        row.update({key:str(val) for key,val in data.items()})
        records[name].append(row)
    def ci() -> dict[str,str]:
        val=Decimal(value())
        return dict(value=str(val),estimate=str(val),ci_low=str(val-Decimal('.03456')),ci_high=str(val+Decimal('.04567')))
    for fi,f in enumerate(tables.FAMILIES):
        add('checkpoint_h_verdicts.csv',model=f,repetition=f+'-rep-s',translation=f+'-tr-s',verdict=f+'-verdict-s',
            s1_repetition=f+'-rep-s1',s1_translation=f+'-tr-s1',s1_verdict=f+'-verdict-s1')
        for ti,fmt in enumerate(tables.FORMATS):
            add('E2-E5_e5.csv',source='primary',model=f,format=fmt,metric='max_LA',target_lang='en',layer=7+fi+ti,n_concepts=1535,**ci())
            for variant in ('s','s1'):
                for metric in ('h1_q','h1_ratio','h2_band_ratio'):
                    add('E2-E5_rules.csv',model=f,format=fmt,condition='diac',fewshot_set=1,item_set='cf729',metric_variant=variant,metric=metric,**ci())
                for k,metric in enumerate(('L_concept_10','L_peak','L_cross')):
                    add('E2-E5_landmarks.csv',model=f,format=fmt,condition='diac',fewshot_set=1,item_set='cf729',metric_variant=variant,
                        metric=metric,value='' if f=='qwen' and fmt=='translation' and metric=='L_cross' else 8+fi+ti+k+(variant=='s1'),n_layers=34+fi)
            for band in ('primary','slope_sensitivity','ops37_as_written'):
                for direction in ('d_vi_en','v_vi'):
                    add('E7_band.csv',model=f,format=fmt,item_set='cf729',direction=direction,band=band,component='share',blocks='2;4;7',
                        included_blocks='2;7' if band=='ops37_as_written' else '2;4;7',requested_size=3,**ci())
            for component in ('attn','mlp'):
                add('E7_band.csv',model=f,format=fmt,item_set='cf729',direction='d_vi_en',band='primary',component=component,
                    sign='-1.0' if component=='mlp' else '1.0',**ci())
        for setting in ('k0.5','k1','k2','random'):
            peak={'gemma':18,'qwen':24,'llama':14}[f]
            last={'gemma':34,'qwen':36,'llama':28}[f]
            rate=str(Decimal('.70')+Decimal(fi)/100+Decimal(len(records['E8_rates.csv']))/10000)
            for layer,en in ((1,'0.12345'),(peak,rate),(last,rate)):
                add('E8_rates.csv',family=f,format='translation',setting=setting,layer=layer,en_rate=en,damage_count=11+layer,n=107+layer)
                fields=ci(); fields['mean_delta']=fields['value']
                add('E8_delta.csv',family=f,format='translation',setting=setting,layer=layer,**fields)
        fields=ci(); fields['mean_delta']=fields['value']
        add('E8_delta.csv',family=f,format='translation',setting='baseline',layer=0,**fields)
        add('E6_dzh.csv',model=f,format='pooled',instrument='D_zh',**ci())
        fields=ci()
        add('E6_model_terms.csv',model=f,instrument='rise_zh_interaction',term='interaction',
            bootstrap_ci_low=fields['ci_low'],bootstrap_ci_high=fields['ci_high'],wald_ci_low=str(Decimal(fields['estimate'])-Decimal('.01234')),
            wald_ci_high=str(Decimal(fields['estimate'])+Decimal('.02345')),**fields)
        for inst in ('D_zh','rise_zh_interaction'):
            add('E6_holm.csv',model=f,instrument=inst,p_holm=value(),reject_0_05=str(fi==1))
            add('E6_matched.csv',model=f,instrument=inst,H3_unresolved=str(fi!=1),**ci())
        for fmt in (*tables.FORMATS,'cloze'):
            for scope in ('paired_main','paired_cf729'):
                for variant,quantity in (('s','Q_H1'),('s1','Q_H1'),('s','H2_band_ratio'),('s','max_LA_en')):
                    add('E9.csv',model=f,format=fmt,scope=scope,condition='nodiac_minus_diac',metric_variant=variant,quantity=quantity,**ci())
                for j,condition in enumerate(('diac','nodiac')):
                    add('E9.csv',model=f,format=fmt,scope=scope,condition=condition,metric_variant='s',quantity='A10_vi_rate',
                        a10_vi_n=31+fi+j,a10_denominator=103+fi+j)
        for label in ('base/'+f,'registered/'+f):
            for variant in ('vi','zh_nospace'):
                add('tables/tableA_tokenization.csv',label=label,family=f,target_variant=variant,mean_content_tokens=value(),
                    continuation_share=value(),usable_count='' if variant=='vi' else 100+fi)
    for set_number in range(2,7):
        for i,(f,fmt) in enumerate((f,t) for f in tables.FAMILIES for t in tables.FORMATS):
            add('R1_set_outcomes.csv',model=f,format=fmt,fewshot_set=set_number,Outcome='same' if i<set_number else 'different',
                s1_outcome='same' if i==set_number else 'different',set1_outcome='same',set1_s1_outcome='same')
    for vi,variant in enumerate(('R4_raw_s','R6_bt_pass_strict_v1','R6_external_attested','R6_all_three_nllb','R6_min_surface_dist_ge_060','R7_competent','R9_main','R11_excluded')):
        for i,(f,fmt) in enumerate((f,t) for f in tables.FAMILIES for t in tables.FORMATS):
            add('R_battery_e4.csv',variant=variant,model=f,format=fmt,Outcome_s='same' if i<=vi%6 else 'different',
                Outcome_s1='same' if i==vi%6 else 'different',primary_Outcome='same',primary_s1_outcome='same')
    for name,rows in records.items():
        path=repo/'reports'/name; path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for r in rows for k in r)),lineterminator='\n')
            writer.writeheader(); writer.writerows(rows)


@pytest.mark.parametrize(('source','mode','expected'),[
    ('1.245','sig','1.24'),('1.255','sig','1.26'),('-1.245','sig','-1.24'),('-1.255','sig','-1.26'),
    ('9.995','sig','10.0'),('99.95','sig','100'),('0.001245','sig','0.00124'),('0','sig','0.00'),
    ('','sig','—'),('NaN','sig','—'),('34.0','int','34'),('0.125','depth','0.12'),('0.135','depth','0.14'),
])
def test_rounding(source: str, mode: str, expected: str) -> None:
    assert tables.rounded(source,mode)==expected


def test_all_tables_preserve_distinct_sources_and_verify(tmp_path: Path) -> None:
    synthetic_sources(tmp_path)
    sheets=tables.generate(tmp_path,committed_only=False)
    assert [s.name for s in sheets]==['T1','T2','T3','T4','T5','T6','T7','TA']
    assert [len(s.rows) for s in sheets]==[6,6,12,3,6,13,18,6]
    assert [row['Variant'].text for row in sheets[5].rows[:5]] == [
        f'R1 set {k}: cells matching set 1' for k in range(2,7)]
    results=tables.verify(tmp_path,committed_only=False)
    assert all(r['fails']==0 for r in results),results
    for sheet in sheets:
        assert sheet.sidecar()
        for row in sheet.sidecar():
            assert row['source_file'] in tables.KEYS
            assert row['source_column']
        text=(tmp_path/'reports/tables'/f'{sheet.name}.tex').read_text()
        assert '\\toprule' in text and '\\midrule' in text and '\\bottomrule' in text
        assert '\\caption' not in text and '\\usepackage' not in text and '\\hline' not in text
    assert '—' in (tmp_path/'reports/tables/T2.md').read_text()
    assert '—' in (tmp_path/'reports/tables/TA.md').read_text()
    t6=sheets[5]
    assert t6.rows[0]['Matching cells s'].text=='2'
    assert t6.rows[0]['Matching cells s1'].text=='1'
    # Same decimal quantum for source CI bounds as for their point estimate.
    cell=sheets[0].rows[0]['Q_H1 s [CI]']
    assert isinstance(cell,tables.Cell)
    assert cell.atoms[1]['quantum']==str(tables.quantum(Decimal(cell.atoms[0]['refs'][0]['source_value'])))
    # Independent semantic checks: the first displayed cells point to the named
    # instrument, variant, direction and scope rather than a neighboring row.
    contracts = [
        ('T1', 'Q_H1 s1 [CI]', 'E2-E5_rules.csv', {'metric':'h1_q','metric_variant':'s1','item_set':'cf729'}),
        ('T2', 'L_peak s1 block (depth)', 'E2-E5_landmarks.csv', {'metric':'L_peak','metric_variant':'s1'}),
        ('T3', 'Delta at peak [CI]', 'E8_delta.csv', {'setting':'k0.5','format':'translation','layer':'18'}),
        ('T4', 'Section 7.1 [Wald CI]', 'E6_model_terms.csv', {'instrument':'rise_zh_interaction'}),
        ('T5', 'Primary share v_vi [CI]', 'E7_band.csv', {'direction':'v_vi','band':'primary','component':'share'}),
        ('T6', 'Matching cells s1', 'R1_set_outcomes.csv', {'fewshot_set':'2'}),
        ('T7', 'Q_H1 difference s1 [CI]', 'E9.csv', {'scope':'paired_main','condition':'nodiac_minus_diac','metric_variant':'s1','quantity':'Q_H1'}),
        ('TA', 'mean_content_tokens', 'tables/tableA_tokenization.csv', {'label':'base/gemma','target_variant':'vi'}),
    ]
    for name, column, filename, filters in contracts:
        sheet = next(sheet for sheet in sheets if sheet.name == name)
        cell = sheet.rows[0][column]
        assert isinstance(cell, tables.Cell)
        ref = cell.atoms[0]['refs'][0]
        assert ref['source_file'] == filename
        assert all(ref['row_keys'][key] == value for key, value in filters.items())
    assert sheets[3].rows[0]['Section 7.1 [Wald CI]'].atoms[1]['refs'][0]['source_column'] == 'wald_ci_low'


def test_peak_lookup_earliest_tie() -> None:
    rows=[tables.SourceRow('E8_rates.csv',dict(layer=str(layer),en_rate=str(value))) for layer,value in ((24,.9),(18,.9),(3,.2))]
    assert tables.peak_lookup(rows).data['layer']=='18'


def test_verify_rejects_altered_display_and_ambiguous_key(tmp_path: Path, monkeypatch: Any) -> None:
    synthetic_sources(tmp_path)
    tables.generate(tmp_path,committed_only=False)
    path=tmp_path/'reports/tables/T3.md'
    path.write_text(path.read_text().replace('0.700','0.999',1))
    # Alter a real displayed value even if the fixture's rate formatting changes.
    text=path.read_text(); lines=text.splitlines(); cells=lines[2].split('|'); cells[3]=' 9.99 '; lines[2]='|'.join(cells)
    path.write_text('\n'.join(lines)+'\n')
    results=tables.verify(tmp_path,committed_only=False)
    assert next(r for r in results if r['table']=='T3')['fails']>0
    original_verify = tables.verify
    monkeypatch.setattr(tables, 'verify', lambda repo: original_verify(repo, committed_only=False))
    assert tables.main(['--repo', str(tmp_path), '--verify']) == 1
    # Duplicate source rows must also fail the exact key-resolution requirement.
    source=tmp_path/'reports/E6_holm.csv'
    content=source.read_text().splitlines(); source.write_text('\n'.join([*content,content[1]])+'\n')
    with pytest.raises(ValueError,match='matches=2'):
        original_verify(tmp_path,committed_only=False)
