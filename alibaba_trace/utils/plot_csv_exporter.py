import os
import pandas as pd
import matplotlib.figure

_original_savefig = matplotlib.figure.Figure.savefig

def _save_csv_alongside_png(self, *args, **kwargs):
    # Save the PNG first.
    ret = _original_savefig(self, *args, **kwargs)
    
    # Get the output file name.
    try:
        filename = kwargs.get('fname', args[0] if args else None)
        if filename is None:
            return ret
            
        filename = str(filename)
        if not filename.endswith('.png'):
            return ret
            
        csv_filename = filename.replace('.png', '.csv')
        
        # Export X and Y values from the figure.
        data = {}
        for i, ax in enumerate(self.axes):
            # Save line plots.
            for j, line in enumerate(ax.get_lines()):
                label = line.get_label()
                if not label or label.startswith('_'):
                    label = f"ax{i}_line{j}"
                data[f"{label}_x"] = line.get_xdata()
                data[f"{label}_y"] = line.get_ydata()
                
            # Save bar charts.
            for j, container in enumerate(ax.containers):
                label = container.get_label()
                if not label or label.startswith('_'):
                    label = f"ax{i}_bar{j}"
                try:
                    data[f"{label}_x"] = [rect.get_x() + rect.get_width() / 2.0 for rect in container.patches]
                    data[f"{label}_y"] = [rect.get_height() for rect in container.patches]
                except Exception:
                    pass
                    
            # Save shaded areas, such as fill_between.
            for j, coll in enumerate(ax.collections):
                label = coll.get_label()
                if not label or label.startswith('_'):
                    label = f"ax{i}_poly{j}"
                try:
                    paths = coll.get_paths()
                    if paths:
                        verts = paths[0].vertices
                        data[f"{label}_x"] = verts[:, 0]
                        data[f"{label}_y"] = verts[:, 1]
                except Exception:
                    pass
                    
        if data:
            df = pd.DataFrame({k: pd.Series(v) for k, v in data.items()})
            df.to_csv(csv_filename, index=False)
            print(f" \033[92m[Data Exported]\033[0m CSV coordinates saved to -> {csv_filename}")
            
    except Exception as e:
        print(f"  [Warning] Could not export CSV for plot: {e}")
        
    return ret

def hook_matplotlib():
    """Call this to automatically intercept matplotlib's savefig and dump CSVs."""
    if matplotlib.figure.Figure.savefig is not _save_csv_alongside_png:
        matplotlib.figure.Figure.savefig = _save_csv_alongside_png
        print("Hooked matplotlib.savefig to automatically export plot CSVs.")

hook_matplotlib()
