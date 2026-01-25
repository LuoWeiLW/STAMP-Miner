# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from Bio.PDB import PDBParser, is_aa

@dataclass
class AtomRec:
    chain: str
    resseq: int
    icode: str
    resname: str
    atom_name: str
    element: str
    xyz: np.ndarray

def _element_from_atom_name(atom_name: str) -> str:
    s = atom_name.strip()
    if not s:
        return ""
    e = s[0].upper()
    if len(s) >= 2 and s[1].islower():
        e = (s[0] + s[1]).title()
    return e

def load_structure(pdb_path: Path):
    parser = PDBParser(QUIET=True)
    return parser.get_structure("complex", str(pdb_path))

def split_protein_ligand(structure, protein_chain: str, ligand_chain: str,
                         ligand_resnames: Optional[set]=None) -> Tuple[List[AtomRec], List[AtomRec]]:
    prot_atoms: List[AtomRec] = []
    lig_atoms: List[AtomRec] = []
    for model in structure:
        for chain in model:
            if chain.id not in {protein_chain, ligand_chain}:
                continue
            for res in chain:
                rn = res.resname.strip()
                aa = is_aa(res, standard=True) or rn in {"MSE"}
                for atom in res:
                    xyz = atom.coord.astype(float)
                    element = getattr(atom, "element", "").strip() or _element_from_atom_name(atom.get_name())
                    rec = AtomRec(chain.id, int(res.id[1]), str(res.id[2]).strip(), rn,
                                  atom.get_name().strip(), element, xyz)
                    if chain.id == protein_chain and aa:
                        prot_atoms.append(rec)
                    elif chain.id == ligand_chain:
                        if ligand_resnames is None or rn in ligand_resnames or (not aa):
                            lig_atoms.append(rec)
    return prot_atoms, lig_atoms

def residue_key(a: AtomRec) -> Tuple[str,int,str,str]:
    return (a.chain, a.resseq, a.icode, a.resname)

def group_by_residue(atoms: List[AtomRec]) -> Dict[Tuple[str,int,str,str], List[AtomRec]]:
    d: Dict[Tuple[str,int,str,str], List[AtomRec]] = {}
    for a in atoms:
        d.setdefault(residue_key(a), []).append(a)
    return d

def min_distance(A: np.ndarray, B: np.ndarray) -> float:
    d = A[:,None,:] - B[None,:,:]
    return float(np.sqrt(np.min(np.sum(d*d, axis=-1))))

def residue_min_distance(resA: List[AtomRec], resB: List[AtomRec]) -> float:
    A = np.vstack([a.xyz for a in resA])
    B = np.vstack([a.xyz for a in resB])
    return min_distance(A,B)

def ring_centroid_and_normal_trp(res_atoms: List[AtomRec]):
    ring_atoms = {"CG","CD1","CD2","NE1","CE2","CE3","CZ2","CZ3","CH2"}
    pts = [a.xyz for a in res_atoms if a.atom_name in ring_atoms]
    if len(pts) < 3:
        pts = [a.xyz for a in res_atoms if a.element != "H"]
    P = np.vstack(pts)
    c = P.mean(axis=0)
    X = P - c
    import numpy.linalg as LA
    _, _, vt = LA.svd(X, full_matrices=False)
    n = vt[-1]
    n = n / (LA.norm(n) + 1e-12)
    return c, n
