"""Copy committed aggregate results into rounded tables with verifiable source atoms."""
from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
from pathlib import Path
from typing import Any, Sequence

FAMILIES = ("gemma", "qwen", "llama")
FORMATS = ("repetition", "translation")
KEYS = {
    "E2-E5_rules.csv": ("model", "format", "condition", "fewshot_set", "target_lang", "item_set", "metric_variant", "metric"),
    "E2-E5_e5.csv": ("source", "model", "format", "metric", "target_lang", "layer"),
    "checkpoint_h_verdicts.csv": ("model",),
    "E2-E5_landmarks.csv": ("model", "format", "condition", "fewshot_set", "item_set", "metric_variant", "metric"),
    "E8_rates.csv": ("family", "format", "setting", "layer"),
    "E8_delta.csv": ("family", "format", "setting", "layer"),
    "E6_dzh.csv": ("model", "format", "instrument"),
    "E6_model_terms.csv": ("model", "instrument", "term"),
    "E6_holm.csv": ("model", "instrument"),
    "E6_matched.csv": ("model", "instrument"),
    "E7_band.csv": ("model", "format", "item_set", "direction", "band", "component"),
    "R1_set_outcomes.csv": ("model", "format", "fewshot_set"),
    "R_battery_e4.csv": ("variant", "model", "format"),
    "E9.csv": ("model", "format", "scope", "condition", "metric_variant", "quantity"),
    "tables/tableA_tokenization.csv": ("label", "family", "target_variant"),
}
FILTERS = {
    "T1": "Six Checkpoint-H cells; diac, cf729 rule rows, s/s1; E5 primary max_LA en rows retain their source n_concepts scope.",
    "T2": "diac, cf729; s/s1; L_concept_10, L_peak, L_cross; formats with cf729 source rows (repetition/translation; no cf729 cloze rows).",
    "T3": "Translation; k0.5/k1/k2/random; earliest maximum EN rate per setting, k1 blocks asserted 18/24/14; delta and DAMAGE at that block; baseline delta at block 0.",
    "T4": "Pooled D_zh and rise_zh_interaction; both Holm instruments and matched-set estimates/flags.",
    "T5": "cf729; primary, slope_sensitivity, ops37_as_written; d_vi_en shares and signs; primary v_vi share.",
    "T6": "R1 sets 2–6, six repetition/translation cells each; stated outcomes compared to set 1. R4/R6×4/R7/R9/R11 outcomes compared to stated primary outcomes; six cells each.",
    "T7": "paired_main/paired_cf729, nodiac_minus_diac, s/s1 Q_H1, s H2 band ratio and max_LA_en; s A10 diac/nodiac count ratios; descriptive cloze separate.",
    "TA": "Base-model label rows only; five requested columns copied from Table A.",
}
DISPLAY_RULE = "3 significant digits, decimal half-even ties; CI bounds use the point's decimal places; layers/counts integer; depth=layer/N to 2 decimals; — for NA. Only requested lookups, count ratios, signs, block-list counts and stated-outcome tallies precede display rounding."


def family(value: str) -> str:
    """Map model identifiers to the fixed presentation order."""
    return next((f for f in FAMILIES if f in value.lower()), value)


def missing(value: str) -> bool:
    """Recognize source NA without imputing it."""
    return value.strip().lower() in {"", "nan", "na", "none", "null"}


def quantum(value: Decimal) -> Decimal:
    """Return the quantum for three significant digits, including rounding carry."""
    if not value:
        return Decimal(".01")
    q = Decimal(1).scaleb(value.adjusted() - 2)
    rounded = value.quantize(q, rounding=ROUND_HALF_EVEN)
    return Decimal(1).scaleb(rounded.adjusted() - 2) if rounded else q


def rounded(value: str | Decimal, mode: str = "sig", q: str = "") -> str:
    """Format exact source decimals; ties use ROUND_HALF_EVEN, including negatives."""
    if isinstance(value, str) and missing(value):
        return "—"
    if mode == "text":
        return str(value)
    d = Decimal(value)
    if not d.is_finite():
        return "—"
    if mode == "int":
        if d != d.to_integral_value():
            raise ValueError(f"Non-integral layer/count: {value}")
        return str(int(d))
    unit = Decimal(q) if q else Decimal(".01") if mode == "depth" else quantum(d)
    result = d.quantize(unit, rounding=ROUND_HALF_EVEN)
    if result == 0:
        result = abs(result)
    return format(result, f".{max(0,-unit.as_tuple().exponent)}f")


@dataclass(frozen=True)
class SourceRow:
    """One source row and its stable unique key."""
    file: str
    data: dict[str, str]

    def ref(self, column: str) -> dict[str, Any]:
        """Reference the full-precision source string and its row key."""
        return {"source_file": self.file, "row_keys": {k:self.data[k] for k in KEYS[self.file]},
                "source_column": column, "source_value": self.data[column]}


@dataclass
class Sources:
    """Read only allow-listed committed CSVs, or isolated synthetic fixtures."""
    repo: Path
    committed_only: bool = True
    cache: dict[str, list[SourceRow]] = field(default_factory=dict)

    def rows(self, name: str, **filters: str) -> list[SourceRow]:
        """Select exact source rows, with model IDs mapped only for family filtering."""
        if name not in KEYS:
            raise ValueError(f"Source not allow-listed: {name}")
        if name not in self.cache:
            if self.committed_only:
                raw = subprocess.check_output(["git","show",f"HEAD:reports/{name}"],cwd=self.repo).decode()
            else:
                raw = (self.repo / "reports" / name).read_text()
            self.cache[name] = [SourceRow(name, r) for r in csv.DictReader(io.StringIO(raw))]
        return [r for r in self.cache[name] if all(
            family(r.data.get("model", r.data.get("family",""))) == v if k == "family_key"
            else r.data.get(k) == v for k,v in filters.items())]

    def one(self, name: str, **filters: str) -> SourceRow:
        """Require a unique row; missing/ambiguous source selections fail closed."""
        found = self.rows(name, **filters)
        if len(found) != 1:
            raise ValueError(f"{name} {filters}: matches={len(found)}")
        return found[0]


def evaluate(operation: str, refs: list[dict[str, Any]], parameter: str = "") -> str:
    """Apply only task-specified display operations to raw source strings."""
    values = [r["source_value"] for r in refs]
    if operation == "copy":
        return values[0]
    if operation == "ratio":
        if any(missing(v) for v in values): return ""
        with localcontext() as context:
            context.prec = 50
            return str(Decimal(values[0])/Decimal(values[1]))
    if operation == "sign":
        if missing(values[0]): return ""
        return "+" if Decimal(values[0]) > 0 else "−" if Decimal(values[0]) < 0 else "0"
    if operation == "block_count":
        return "" if missing(values[0]) else str(len(values[0].split(";")))
    if operation == "block_item":
        return "" if missing(values[0]) else values[0].split(";")[int(parameter)]
    if operation == "tally_equal":
        if len(values) % 2: raise ValueError("Unpaired tally operands")
        return str(sum(values[i] == values[i+1] for i in range(0,len(values),2)))
    raise ValueError(operation)


@dataclass
class Cell:
    """Displayed text and the source-backed numeric/category atoms it contains."""
    text: str
    atoms: list[dict[str, Any]] = field(default_factory=list)

    def __add__(self, other: Cell | str) -> Cell:
        other = other if isinstance(other, Cell) else Cell(other)
        return Cell(self.text + other.text, self.atoms + other.atoms)

    def __radd__(self, other: str) -> Cell:
        return Cell(other) + self


def atom(row: SourceRow, column: str, mode: str = "sig", *, operation: str = "copy",
         extra: list[dict[str, Any]] | None = None, parameter: str = "", q: str = "") -> Cell:
    """Create one displayed atom with complete source operands."""
    refs = [row.ref(column), *(extra or [])]
    value = evaluate(operation, refs, parameter)
    display = rounded(value, mode, q)
    return Cell(display,[dict(operation=operation,parameter=parameter,mode=mode,quantum=q,display_value=display,refs=refs)])


def interval(row: SourceRow, value: str = "value", low: str = "ci_low", high: str = "ci_high") -> Cell:
    """Copy an estimate and round its bounds to the point's decimal places."""
    raw = row.data[value]
    if missing(raw):
        return atom(row, value)
    q = str(quantum(Decimal(raw))) if not missing(raw) else ".01"
    return atom(row,value) + " [" + atom(row,low,q=q) + ", " + atom(row,high,q=q) + "]"


def blocks(row: SourceRow, column: str = "blocks") -> Cell:
    """Display each requested block as a separate source-backed integer atom."""
    if missing(row.data[column]): return atom(row,column,"text")
    result=Cell("")
    for i,_ in enumerate(row.data[column].split(";")):
        result = result + (", " if i else "") + atom(row,column,"int",operation="block_item",parameter=str(i))
    return result


def peak_lookup(rows: Sequence[SourceRow]) -> SourceRow:
    """OPS-27 lookup: maximum EN rate, earliest block on ties."""
    return sorted(rows,key=lambda r:(-Decimal(r.data["en_rate"]),int(r.data["layer"])))[0]


def block_depth(row: SourceRow, column: str, denominator: SourceRow, denominator_column: str) -> Cell:
    """Copy a block and divide only by its source-backed layer count."""
    if missing(row.data[column]):
        return atom(row, column, "int")
    return atom(row,column,"int") + " (" + atom(row,column,"depth",operation="ratio",extra=[denominator.ref(denominator_column)]) + ")"


@dataclass
class Sheet:
    """A table with stable row labels and a full atom-level provenance sidecar."""
    name: str
    rows: list[dict[str, Cell | str]] = field(default_factory=list)

    def add(self, **cells: Cell | str) -> None:
        """Append an ordered display row."""
        self.rows.append(cells)

    def sidecar(self) -> list[dict[str,str]]:
        """List every operand of every displayed atom, with its source value."""
        result=[]
        for i,row in enumerate(self.rows,1):
            for column,cell in row.items():
                if not isinstance(cell,Cell): continue
                for j,part in enumerate(cell.atoms,1):
                    for k,ref in enumerate(part["refs"],1):
                        result.append(dict(table=self.name,display_row=str(i),display_column=column,atom=str(j),operand=str(k),
                                           operation=part["operation"],parameter=part["parameter"],mode=part["mode"],quantum=part["quantum"],
                                           display_value=part["display_value"],cell_text=cell.text,source_file=ref["source_file"],
                                           row_keys=json.dumps(ref["row_keys"],sort_keys=True,separators=(",",":")),
                                           source_column=ref["source_column"],source_value=ref["source_value"]))
        return result


def make_tables(source: Sources) -> list[Sheet]:
    """Build T1–T7 and TA from copied rows and explicitly requested operations."""
    sheets=[Sheet(n) for n in ("T1","T2","T3","T4","T5","T6","T7","TA")]
    t1,t2,t3,t4,t5,t6,t7,ta=sheets
    for f in FAMILIES:
        verdict=source.one("checkpoint_h_verdicts.csv",family_key=f)
        for fmt in FORMATS:
            e5=source.one("E2-E5_e5.csv",family_key=f,format=fmt,source="primary",metric="max_LA",target_lang="en")
            data={"Model":f,"Format":fmt,"max LA_en primary/main [CI] (block)":interval(e5)+" ("+atom(e5,"layer","int")+")"}
            for variant in ("s","s1"):
                for label,metric in (("Q_H1","h1_q"),("H1 ratio","h1_ratio"),("H2 band ratio","h2_band_ratio")):
                    r=source.one("E2-E5_rules.csv",family_key=f,format=fmt,condition="diac",item_set="cf729",metric_variant=variant,metric=metric)
                    data[f"{label} {variant} [CI]"]=interval(r)
                data[f"Outcome {variant}"]=atom(verdict,fmt if variant=="s" else "s1_"+fmt,"text")
            data["Model verdict s"]=atom(verdict,"verdict","text")
            data["Model verdict s1"]=atom(verdict,"s1_verdict","text")
            t1.add(**data)
        for fmt in ("repetition","translation","cloze"):
            if not source.rows("E2-E5_landmarks.csv", family_key=f, format=fmt, condition="diac", item_set="cf729"):
                continue
            data={"Model":f,"Format":fmt}
            for variant in ("s","s1"):
                for metric in ("L_concept_10","L_peak","L_cross"):
                    r=source.one("E2-E5_landmarks.csv",family_key=f,format=fmt,condition="diac",item_set="cf729",metric_variant=variant,metric=metric)
                    data[f"{metric} {variant} block (depth)"]=block_depth(r,"value",r,"n_layers")
            t2.add(**data)
        for setting in ("k0.5","k1","k2","random"):
            rates=source.rows("E8_rates.csv",family_key=f,format="translation",setting=setting)
            peak=peak_lookup(rates)
            if setting=="k1":
                assert int(peak.data["layer"])=={"gemma":18,"qwen":24,"llama":14}[f],f"OPS-27 k1 peak mismatch: {f}"
            last=sorted(rates,key=lambda r:int(r.data["layer"]))[-1]
            delta=source.one("E8_delta.csv",family_key=f,format="translation",setting=setting,layer=peak.data["layer"])
            baseline=source.one("E8_delta.csv",family_key=f,format="translation",setting="baseline",layer="0")
            t3.add(Model=f,Setting=setting,**{"Peak EN rate":atom(peak,"en_rate"),"Block (depth)":block_depth(peak,"layer",last,"layer"),
                    "DAMAGE at peak":atom(peak,"damage_count",operation="ratio",extra=[peak.ref("n")]),
                    "Delta at peak [CI]":interval(delta,"mean_delta"),"Unsteered delta [CI]":interval(baseline,"mean_delta")})
        d=source.one("E6_dzh.csv",family_key=f,format="pooled",instrument="D_zh")
        term=source.one("E6_model_terms.csv",family_key=f,instrument="rise_zh_interaction")
        data={"Model":f,"D_zh pooled [bootstrap CI]":interval(d,"estimate"),
              "Section 7.1 [bootstrap CI]":interval(term,"estimate","bootstrap_ci_low","bootstrap_ci_high"),
              "Section 7.1 [Wald CI]":interval(term,"estimate","wald_ci_low","wald_ci_high")}
        for inst,label in (("D_zh","D_zh"),("rise_zh_interaction","Section 7.1")):
            h=source.one("E6_holm.csv",family_key=f,instrument=inst)
            matched=source.one("E6_matched.csv",family_key=f,instrument=inst)
            data[label+" Holm p"]=atom(h,"p_holm")
            data[label+" rejected"]=atom(h,"reject_0_05","text")
            data[label+" matched [CI]"]=interval(matched,"estimate")
            data[label+" H3 unresolved"]=atom(matched,"H3_unresolved","text")
        t4.add(**data)
        for fmt in FORMATS:
            filters=dict(family_key=f,format=fmt,item_set="cf729")
            primary=source.one("E7_band.csv",**filters,direction="d_vi_en",band="primary",component="share")
            data={"Model":f,"Format":fmt,"Primary blocks":blocks(primary)}
            for comp in ("attn","mlp"):
                r=source.one("E7_band.csv",**filters,direction="d_vi_en",band="primary",component=comp)
                data["S_"+comp+" sign d_hat"]=atom(r,"sign","text",operation="sign")
            data["Primary share d_hat [CI]"]=interval(primary,"estimate")
            vi=source.one("E7_band.csv",**filters,direction="v_vi",band="primary",component="share")
            data["Primary share v_vi [CI]"]=interval(vi,"estimate")
            for band in ("slope_sensitivity","ops37_as_written"):
                r=source.one("E7_band.csv",**filters,direction="d_vi_en",band=band,component="share")
                data[band+" share d_hat [CI]"]=interval(r,"estimate")
                data[band+" included/requested"]=atom(r,"included_blocks","int",operation="block_count")+"/"+atom(r,"requested_size","int")
            t5.add(**data)
        for scope in ("paired_main","paired_cf729"):
            for fmt in ("repetition","translation","cloze"):
                filters=dict(family_key=f,format=fmt,scope=scope,condition="nodiac_minus_diac")
                data={"Model":f,"Format":fmt,"Scope":scope}
                for variant in ("s","s1"):
                    r=source.one("E9.csv",**filters,metric_variant=variant,quantity="Q_H1")
                    data[f"Q_H1 difference {variant} [CI]"]=interval(r)
                for quantity in ("H2_band_ratio","max_LA_en"):
                    r=source.one("E9.csv",**filters,metric_variant="s",quantity=quantity)
                    data[quantity+" difference s [CI]"]=interval(r)
                a=source.one("E9.csv",family_key=f,format=fmt,scope=scope,condition="diac",metric_variant="s",quantity="A10_vi_rate")
                b=source.one("E9.csv",family_key=f,format=fmt,scope=scope,condition="nodiac",metric_variant="s",quantity="A10_vi_rate")
                data["A10 VI rate diac → nodiac"]=atom(a,"a10_vi_n",operation="ratio",extra=[a.ref("a10_denominator")])+" → "+atom(b,"a10_vi_n",operation="ratio",extra=[b.ref("a10_denominator")])
                t7.add(**data)
    for set_number in range(2,7):
        rows=source.rows("R1_set_outcomes.csv",fewshot_set=str(set_number))
        if len(rows)!=6: raise ValueError(f"R1 set {set_number}: expected six cells")
        data={"Variant":Cell("R1 set ")+atom(rows[0],"fewshot_set","int")+": cells matching set 1"}
        for variant,col,reference in (("s","Outcome","set1_outcome"),("s1","s1_outcome","set1_s1_outcome")):
            refs=[ref for r in rows for ref in (r.ref(col),r.ref(reference))]
            data[f"Matching cells {variant}"]=atom(rows[0],col,"int",operation="tally_equal",extra=refs[1:])
        t6.add(**data)
    for variant in ("R4_raw_s","R6_bt_pass_strict_v1","R6_external_attested","R6_all_three_nllb","R6_min_surface_dist_ge_060","R7_competent","R9_main","R11_excluded"):
        rows=source.rows("R_battery_e4.csv",variant=variant)
        if len(rows)!=6: raise ValueError(f"{variant}: expected six cells")
        data={"Variant":variant}
        for score,col,refcol in (("s","Outcome_s","primary_Outcome"),("s1","Outcome_s1","primary_s1_outcome")):
            refs=[ref for r in rows for ref in (r.ref(col),r.ref(refcol))]
            data[f"Matching cells {score}"]=atom(rows[0],col,"int",operation="tally_equal",extra=refs[1:])
        t6.add(**data)
    for r in source.rows("tables/tableA_tokenization.csv"):
        if r.data["label"].startswith("base/"):
            ta.add(family=r.data["family"],target_variant=r.data["target_variant"],mean_content_tokens=atom(r,"mean_content_tokens"),
                   continuation_share=atom(r,"continuation_share"),usable_count=atom(r,"usable_count","int"))
    return sheets


def tex_escape(text: str) -> str:
    """Escape table text without requiring any LaTeX package beyond booktabs."""
    replacements={"\\":r"\textbackslash{}","_":r"\_","%":r"\%","&":r"\&","#":r"\#","{":r"\{","}":r"\}","$":r"\$",
                  "→":r"$\to$","−":"-","—":"---"}
    return "".join(replacements.get(c,c) for c in text)


def source_note(sheet: Sheet) -> str:
    """Return a single factual source line, without caption prose."""
    names=sorted({r["source_file"] for r in sheet.sidecar()})
    scope = " E5 max LA_en: primary/main scope; rule quantities: cf729." if sheet.name == "T1" else ""
    return "Sources: " + "; ".join("reports/"+n for n in names)+"."+scope


def render(sheet: Sheet) -> tuple[str,str]:
    """Render the same cells in Markdown and booktabs-only LaTeX."""
    headers=list(sheet.rows[0])
    values=[[c.text if isinstance(c,Cell) else c for c in row.values()] for row in sheet.rows]
    md=["| "+" | ".join(headers)+" |","| "+" | ".join("---" for _ in headers)+" |"]
    md += ["| "+" | ".join(row)+" |" for row in values]
    md += ["",source_note(sheet)]
    tex=[r"\begin{tabular}{"+"l"*len(headers)+"}",r"\toprule"," & ".join(map(tex_escape,headers))+r" \\",r"\midrule"]
    tex += [" & ".join(map(tex_escape,row))+r" \\" for row in values]
    tex += [r"\bottomrule",r"\end{tabular}",r"\par\noindent{\footnotesize "+tex_escape(source_note(sheet))+"}"]
    return "\n".join(md)+"\n","\n".join(tex)+"\n"


def sidecar_bytes(sheet: Sheet) -> str:
    """Serialize the atom provenance deterministically."""
    rows=sheet.sidecar(); stream=io.StringIO(newline="")
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    return stream.getvalue()


def source_commit(repo: Path, source: str) -> str:
    """Return the commit last changing the committed source CSV."""
    return subprocess.check_output(["git","log","-1","--format=%H","HEAD","--","reports/"+source],cwd=repo,text=True).strip()


def generate(repo: Path, *, committed_only: bool = True) -> list[Sheet]:
    """Write eight tables and their provenance, reading committed CSVs by default."""
    sheets=make_tables(Sources(repo,committed_only))
    output=repo/"reports/tables"; (output/"data").mkdir(parents=True,exist_ok=True)
    readme=["# Result tables","",DISPLAY_RULE,""]
    for sheet in sheets:
        md,tex=render(sheet)
        (output/(sheet.name+".md")).write_text(md)
        (output/(sheet.name+".tex")).write_text(tex)
        (output/"data"/(sheet.name+".csv")).write_text(sidecar_bytes(sheet))
        readme += ["## "+sheet.name,"",f"Files: {sheet.name}.md; {sheet.name}.tex; data/{sheet.name}.csv.","", "Filters: "+FILTERS[sheet.name],"", "Display: "+DISPLAY_RULE, ""]
        for name in sorted({r["source_file"] for r in sheet.sidecar()}):
            commit=source_commit(repo,name) if committed_only else "synthetic fixture"
            readme.append(f"Source: reports/{name}; commit {commit}.")
        readme.append("")
    (output/"README.md").write_text("\n".join(readme))
    return sheets


def verify(repo: Path, *, committed_only: bool = True) -> list[dict[str, Any]]:
    """Resolve every sidecar operand and compare actual displays with rounded sources."""
    source=Sources(repo,committed_only); expected=make_tables(source); results=[]
    for sheet in expected:
        failures=[]; output=repo/"reports/tables"
        md,tex=render(sheet)
        for suffix,text in (("md",md),("tex",tex)):
            path=output/(sheet.name+"."+suffix)
            if not path.is_file() or path.read_text()!=text:
                failures.append(suffix+" displayed cells differ from rounded sources")
        path=output/"data"/(sheet.name+".csv")
        if not path.is_file():
            failures.append("sidecar missing"); rows=[]
        else:
            rows=list(csv.DictReader(io.StringIO(path.read_text())))
            if path.read_text()!=sidecar_bytes(sheet): failures.append("sidecar differs from source-backed expected atoms")
        for row in rows:
            try:
                selected=source.one(row["source_file"],**json.loads(row["row_keys"]))
                if selected.data[row["source_column"]]!=row["source_value"]:
                    failures.append("full-precision source string mismatch")
            except (ValueError,KeyError,json.JSONDecodeError) as error:
                failures.append(str(error))
        results.append(dict(table=sheet.name,operands=len(rows),fails=len(failures),reasons=failures[:5]))
    return results


def main(argv: Sequence[str] | None = None) -> int:
    """Generate or verify committed-source result tables, without analysis imports."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config",default="configs/experiments.yaml",help="Compatibility with repository entrypoints; no model/data configuration is loaded.")
    parser.add_argument("--verify",action="store_true")
    args=parser.parse_args(argv)
    if args.verify:
        results=verify(args.repo)
        for result in results:
            print(f"{result['table']}: operands={result['operands']}; FAIL={result['fails']}")
            for reason in result["reasons"]: print("  "+reason)
        return int(any(r["fails"] for r in results))
    for sheet in generate(args.repo): print(f"{sheet.name}: rows={len(sheet.rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
