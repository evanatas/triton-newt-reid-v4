"""Часть 11 — ЕДИНСТВЕННОЕ необратимое вскрытие sealed TK-test. Требует --confirm.

Церемония: verify sha256 манифеста → opened==False → материализовать test-кропы (тот же пайплайн birefnet_label) →
mark_opened → прогнать ЗАМОРОЖЕННЫЕ методы (ядро SIFT+affine rolling+single, MegaDescriptor, DINOv2) на test →
artifacts/final_sealed_test.json. Числа test руками НЕ правятся. test ≤ dev 0.753 — норма (gap обобщения).

Запуск (необратимо!):  python seal_open.py --confirm
"""
from __future__ import annotations
import argparse
import datetime
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
for _p in ("..", "../registry", "../segment", "../spike_lab"):
    sys.path.insert(0, str((HERE / _p).resolve()))
sys.path.insert(0, str(HERE))
from sealed import verify_manifest, mark_opened, MANIFEST     # noqa: E402
import build_crops as BC                                       # noqa: E402
import eval_core as EC                                         # noqa: E402
import baselines as BL                                         # noqa: E402

ART = HERE / "artifacts"
REG = HERE / ".." / "registry"


def assemble_final(opened_date):
    """Собрать финальную KPI-таблицу из baselines_test.json (сравнение) + eval_tk_test.json (open-set)."""
    bl = json.loads((ART / "baselines_test.json").read_text())
    ev = json.loads((ART / "eval_tk_test.json").read_text())
    m = bl["methods"]
    core_roll = m["core_sift_affine"]["rolling"]

    def br1(k):   # rolling R@1 бейзлайна (None если пропущен/не посчитан)
        return round(m[k]["rolling"]["micro_recall"]["1"], 3) if k in m and "rolling" in m[k] else None

    final = {
        "FINAL_KPI_sealed_test": True, "opened_date": opened_date, "goal": 0.75, "v3.0_final": 0.188,
        "n_probe_known": core_roll["n_probe"], "random_top1": bl["random_top1"],
        "core_rolling": {"recall@1": round(core_roll["micro_recall"]["1"], 3), "ci95": core_roll["wilson@1"],
                         "macro@1": round(core_roll["macro_recall"]["1"], 3),
                         "recall@5": round(core_roll["micro_recall"]["5"], 3),
                         "recall@10": round(core_roll["micro_recall"]["10"], 3),
                         "frr@1": round(core_roll["frr"]["1"], 3), "by_interval": core_roll["by_interval"]},
        "core_single_recall@1": round(m["core_sift_affine"]["single"]["micro_recall"]["1"], 3),
        "baselines_rolling_recall@1": {"megadescriptor": br1("megadescriptor"), "dinov2": br1("dinov2")},
        "mcnemar_core_vs_baseline_rolling": {k: m[k]["rolling"].get("mcnemar_vs_core@1")
                                             for k in ("megadescriptor", "dinov2") if k in m and "rolling" in m[k]},
        "open_set_test": ev.get("open_set"),
        "goal_reached": round(core_roll["micro_recall"]["1"], 3) >= 0.75,
        "kpi_framing_F5": "goal_reached судится по rolling micro (накапливаемый каталог); single-ref — нижняя граница "
                          "(единственный прежний снимок). Подавать РАЗДЕЛЬНО; rolling != one-shot.",
        "artifacts": ["baselines_test.json", "eval_tk_test.json"],
    }
    (ART / "final_sealed_test.json").write_text(json.dumps(final, ensure_ascii=False, indent=2))
    return final


def rehearse():
    """F-2: генеральная репетиция БЕЗ вскрытия. (1) все sealed test-файлы существуют и открываются;
    (2) body_mask детерминирован (проверка на DEV-кадре, sealed не трогаем); (3) manifest-append корректен.
    Ничего не материализует, opened НЕ меняет, метрики на test НЕ считает."""
    print("🧪 РЕПЕТИЦИЯ (dry-run, sealed НЕ вскрывается)\n")
    reg = pd.read_csv(REG / "registry.csv")
    spl = pd.read_csv(REG / "splits.csv")[["frame_id", "split_fold"]]
    df = reg.merge(spl, on="frame_id", how="left")
    ok = True

    test = df[df.split_fold == "test"]                  # (1) test-файлы существуют/открываются
    bad = []
    for _, r in test.iterrows():
        p = BC.ROOT / str(r.path_rel)
        if not isinstance(r.path_rel, str) or not p.exists():
            bad.append(("нет файла/NaN", r.frame_id)); continue
        try:
            with BC.Image.open(p) as im:
                im.verify()
        except Exception as e:
            bad.append(("не открывается", r.frame_id, str(e)[:40]))
    print(f"  (1) test-кадров {len(test)}: битых/отсутствующих {len(bad)} {'✅' if not bad else '❌'}")
    for b in bad[:5]:
        print(f"       {b}")
    ok &= not bad

    dev1 = df[df.split_fold == "dev"].iloc[0]            # (2) детерминизм body_mask на DEV-кадре
    img = BC.np.array(BC.Image.open(BC.ROOT / dev1.path_rel).convert("RGB"))
    m1, _, _ = BC.body_mask(img)
    m2, _, _ = BC.body_mask(img)
    det = bool(BC.np.array_equal(m1, m2))
    print(f"  (2) body_mask детерминизм (2 прогона): {'идентично ✅' if det else 'РАЗЛИЧАЕТСЯ ❌'}")
    ok &= det

    man = pd.read_csv(HERE / ".." / "segment" / "crops_manifest.csv")   # (3) manifest-append (в памяти)
    add = pd.DataFrame([{c: (test.iloc[0].frame_id if c == "frame_id" else "") for c in man.columns}])
    merged = pd.concat([man[~man.frame_id.isin(add.frame_id)], add], ignore_index=True)
    app_ok = (len(merged) == len(man) + 1) and (merged.frame_id.duplicated().sum() == 0)
    print(f"  (3) manifest-append (dummy): {'корректно ✅' if app_ok else 'СБОЙ ❌'}")
    ok &= app_ok

    print("\n" + ("✅ РЕПЕТИЦИЯ ЗЕЛЁНАЯ — путь безопасен, можно запускать --confirm" if ok
                  else "❌ РЕПЕТИЦИЯ КРАСНАЯ — НЕ вскрывать, чинить причину"))
    return ok


def main(confirm):
    if not confirm:
        print("ОТКАЗ: вскрытие sealed необратимо. Запусти: python seal_open.py --confirm (или --rehearse для репетиции)")
        return
    reg = pd.read_csv(REG / "registry.csv")
    spl = pd.read_csv(REG / "splits.csv")[["frame_id", "split_fold"]]
    df = reg.merge(spl, on="frame_id", how="left")
    if not verify_manifest(df):
        print("❌ ABORT: sha256 манифеста НЕ совпал — test-сплит изменён с момента запечатывания. НЕ вскрываю.")
        return
    man = json.loads(MANIFEST.read_text())
    if man.get("opened"):
        print(f"❌ ABORT: test УЖЕ вскрыт ({man.get('opened_date')}). Повторное вскрытие запрещено (одноразовость).")
        return
    print(f"✅ sha256 манифеста совпал ({man['sha256'][:12]}…) | opened=False | n_sealed={man['n_sealed']}")
    print("🔓 ВСКРЫВАЮ sealed TK-test (РОВНО ОДИН РАЗ)...\n")

    BC.build_test_only()                                # материализовать кропы теста (тот же пайплайн)
    opened_date = datetime.date.today().isoformat()
    # F-1: прогон ДО пометки opened. Падение внутри не «сжигает» печать (opened остаётся False → можно повторить).
    try:
        print("─ ядро + global-бейзлайны на TEST ─")
        BL.main(fold="test", unseal=True)               # core+Mega+DINOv2 + McNemar -> baselines_test.json
        print("\n─ ядро: метрики честности + open-set(17 new) на TEST ─")
        EC.main("TK", fold="test", unseal=True)         # rolling/single + macro/FRR + open-set -> eval_tk_test.json
        final = assemble_final(opened_date)
    except Exception as e:
        print(f"\n❌ ПРОГОН УПАЛ: {e}")
        print("   Манифест НЕ помечен opened=True — печать НЕ сожжена. Почини причину и запусти снова.")
        raise
    mark_opened(opened_date)                            # F-1: пометка одноразовости — ТОЛЬКО после успешного прогона
    print(f"\n  манифест помечен opened=True ({opened_date})")
    k = final["core_rolling"]
    print("\n" + "=" * 66)
    print("ФИНАЛЬНЫЙ KPI — sealed TK-test (вскрыт РОВНО один раз):")
    print(f"  ЯДРО rolling R@1 = {k['recall@1']} {k['ci95']}  (macro {k['macro@1']}, R@5 {k['recall@5']}, R@10 {k['recall@10']}, FRR@1 {k['frr@1']})")
    print(f"  ядро single-ref R@1 = {final['core_single_recall@1']}  |  random {final['random_top1']}")
    print(f"  бейзлайны rolling R@1: {final['baselines_rolling_recall@1']}")
    print(f"  по интервалам R@1: {k['by_interval']}")
    print(f"  ЦЕЛЬ 0.75: {'ДОСТИГНУТА ✅' if final['goal_reached'] else 'НЕ достигнута'}  |  vs 3.0 финал 0.188")
    print("=" * 66)
    print("✅ artifacts/final_sealed_test.json — числа НЕ правим (финальный KPI)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", action="store_true", help="подтвердить необратимое вскрытие sealed test")
    ap.add_argument("--rehearse", action="store_true", help="генеральная репетиция (dry-run, sealed НЕ вскрывается)")
    a = ap.parse_args()
    rehearse() if a.rehearse else main(a.confirm)
