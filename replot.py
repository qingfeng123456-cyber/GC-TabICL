"""Redraw every returned figure using NumPy and Matplotlib on CPU."""
from pathlib import Path
import argparse
import figures
import study_figures

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--logs',type=Path,required=True,help='Returned run directory')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();root=args.logs/'figures'
    figures.redraw(root/'replot_arrays.npz',args.out)
    evidence=root/'study_evidence.json'
    if evidence.exists():study_figures.render(study_figures.json.loads(evidence.read_text('utf-8')),args.out)
    print('CPU redraw complete:',args.out)
