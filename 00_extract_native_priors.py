# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from core.interaction_priors import (
    extract_native_contacts,
    build_site_profile,
    w136_geometry,
    extract_w136_shell_from_native,
    build_site_profile_from_df
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--complex_pdb", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--protein_chain", default="A")
    ap.add_argument("--ligand_chain", default="G")
    ap.add_argument("--ligand_resnames", default="NAG", help="comma-separated (e.g., NAG)")
    ap.add_argument("--contact_cutoff", type=float, default=4.5)

    ap.add_argument("--w136_chain", default="A")
    ap.add_argument("--w136_resseq", type=int, default=136)
    ap.add_argument("--w136_boost", type=float, default=2.0)

    ap.add_argument("--shell_radius", type=float, default=6.0, help="W136 shell radius in Å (recommended 6.0)")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    lig_res = [x.strip() for x in str(args.ligand_resnames).split(",") if x.strip()]

    # --- Global pocket contacts/profile (as before) ---
    native = extract_native_contacts(Path(args.complex_pdb),
                                     protein_chain=args.protein_chain,
                                     ligand_chain=args.ligand_chain,
                                     ligand_resnames=lig_res,
                                     cutoff=float(args.contact_cutoff))
    df_global = native["contacts_df"]
    df_global.to_csv(out / "native_contacts_residue.csv", index=False)

    w136_id = f"{args.w136_chain}:TRP{int(args.w136_resseq)}"

    # Global profile (keep original filename for downstream compatibility)
    site_global = build_site_profile(df_global, w136_residue_id=w136_id, w136_boost=float(args.w136_boost))
    (out / "site_profile_global.json").write_text(json.dumps(site_global, indent=2), encoding="utf-8")
    # also write legacy name if you want compatibility
    (out / "site_profile.json").write_text(json.dumps(site_global, indent=2), encoding="utf-8")

    # --- W136 shell contacts/profile (new) ---
    df_shell = extract_w136_shell_from_native(Path(args.complex_pdb),
                                              protein_chain=args.protein_chain,
                                              ligand_chain=args.ligand_chain,
                                              ligand_resnames=lig_res,
                                              contact_cutoff=float(args.contact_cutoff),
                                              w136_chain=args.w136_chain,
                                              w136_resseq=int(args.w136_resseq),
                                              shell_radius=float(args.shell_radius))
    df_shell.to_csv(out / "w136_shell_residues.csv", index=False)

    if len(df_shell) > 0:
        site_shell = build_site_profile_from_df(
            df_shell,
            w136_residue_id=w136_id,
            w136_boost=1.0,
            dist_col="min_dist_to_ligand",
            extra_decay_col="min_dist_to_W136"
        )
    else:
        # fallback to global if shell empty (should be rare)
        site_shell = site_global

    (out / "site_profile_w136_shell.json").write_text(json.dumps(site_shell, indent=2), encoding="utf-8")

    # --- W136 geometry ---
    geom = w136_geometry(Path(args.complex_pdb),
                         protein_chain=args.protein_chain,
                         ligand_chain=args.ligand_chain,
                         w136_chain=args.w136_chain,
                         w136_resseq=int(args.w136_resseq),
                         ligand_resnames=lig_res)
    (out / "w136_geometry.json").write_text(json.dumps(geom, indent=2), encoding="utf-8")

    summary = {
        "n_contacts_residues_global": int(len(df_global)),
        "n_contacts_residues_w136_shell": int(len(df_shell)),
        "contact_cutoff": float(args.contact_cutoff),
        "shell_radius": float(args.shell_radius),
        "protein_chain": args.protein_chain,
        "ligand_chain": args.ligand_chain,
        "ligand_resnames": lig_res,
        "w136": w136_id,
        "w136_ok": bool(geom.get("ok", False)),
        "w136_min_atom_dist_to_ligand": geom.get("w136_min_atom_dist_to_ligand", None),
        "w136_min_centroid_dist_to_ligand": geom.get("w136_min_centroid_dist_to_ligand", None),
    }
    (out / "native_ifp_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[OK] Saved priors to: {out}")
    print(f"     - site_profile_global.json")
    print(f"     - site_profile_w136_shell.json")
    print(f"     - w136_shell_residues.csv")

if __name__ == "__main__":
    main()
