from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Vertical, Horizontal
from textual.message import Message
from textual.widgets import DataTable, Footer, Header, Label, ListItem, ListView, Static
from textual_plotext import PlotextPlot

from get_collections_and_files import list_files_and_excluded_vars
import traceback

FILE_RE = re.compile(
    r"^quads_results_(?P<model>.+?)_(?P<date>\d{4}_\d{2}_\d{2})_reference_date_(?P<ref>\d{4}-\d{2})\.pkl$"
)

LEV_NAMES = ["lev", "level", "pressure"]
LAT_NAMES = ["lat", "latitude", "y"]
LON_NAMES = ["lon", "longitude", "x"]
TIME_NAMES = ["time", "valid_time", "datetime"]

DATA_YAML_FILE = Path("/home/sadhika8/JupyterLinks/nobackup/quads/conf/dataserver.yaml")
STRATA_FILE = Path("/home/sadhika8/JupyterLinks/nobackup/quads/conf/strata.yaml")


def shorten(value: Any, width: int = 60) -> str:
    text = str(value)
    if len(text) <= width:
        return text
    return text[: width - 1] + "…"


def parse_file_label(path: Path, root: Path) -> str:
    m = FILE_RE.match(path.name)
    rel = path.relative_to(root)
    if not m:
        return str(rel)

    model = m.group("model").upper()
    date = m.group("date").replace("_", "-")
    ref = m.group("ref")
    return f"{model} | {date} | ref {ref}"


def to_1d_float_list(value: Any) -> list[float]:
    if value is None:
        return []

    if isinstance(value, list):
        seq = value
    elif isinstance(value, tuple):
        seq = list(value)
    elif hasattr(value, "tolist"):
        out = value.tolist()
        if isinstance(out, list):
            seq = out
        else:
            seq = [out]
    else:
        try:
            seq = list(value)
        except TypeError:
            return []

    result: list[float] = []
    for item in seq:
        try:
            result.append(float(item))
        except Exception:
            return []
    return result


def load_strata(strata_file: str | Path):
    with open(strata_file, "r") as f:
        return yaml.safe_load(f)["STRATA"]


def pick_name(ds, names):
    return next((n for n in names if n in ds.coords), None)


class QuantilePlot(PlotextPlot):
    def on_mount(self) -> None:
        self.plt.title("Quantile plot")
        self.plt.grid(True, True)

    def show_message(self, title: str) -> None:
        self.plt.clear_data()
        self.plt.clear_figure()
        self.plt.title(title)
        self.refresh()

    def show_row_plot(
        self,
        x_vals: list[float],
        y_vals: list[float],
        title: str,
        fence_low: float | None = None,
        fence_high: float | None = None,
    ) -> None:
        self.plt.clear_data()
        self.plt.clear_figure()

        if not x_vals or not y_vals:
            self.plt.title("No plottable data")
            self.refresh()
            return

        if len(x_vals) != len(y_vals):
            self.plt.title("q_list and quantile_values have different lengths")
            self.refresh()
            return

        self.plt.title(title)
        self.plt.plot(x_vals, y_vals)

        if fence_low is not None:
            self.plt.plot(x_vals, [fence_low] * len(x_vals))
        if fence_high is not None:
            self.plt.plot(x_vals, [fence_high] * len(x_vals))

        self.refresh()


class PklBrowser(App):
    CSS = """
    Screen {
        layout: horizontal;
    }

    #left {
        width: 40;
        border: solid $accent;
    }

    #right {
        width: 1fr;
    }

    #file_summary {
        height: 10;
        border: solid green;
        padding: 1;
    }

    #row_summary {
        height: 12;
        border: solid yellow;
        padding: 1;
    }

    #preview {
        height: 8;
        border: solid cyan;
    }

    #plot_area {
        height: 1fr;
        layout: horizontal;
    }

    #flagged_panel {
        width: 72;
        border: solid magenta;
    }

    #flagged_title {
        height: 3;
        border: solid magenta;
        padding: 0 1;
    }

    #flagged_table {
        height: 1fr;
        border: solid magenta;
    }

    #plot_right {
        width: 1fr;
        border: solid blue;
    }

    ListView {
        height: 1fr;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "reload_files", "Reload files"),
        ("enter", "activate_current", "Open / Select"),
        ("f", "focus_files", "Files"),
        ("t", "focus_table", "Preview Table"),
        ("g", "focus_flagged_table", "Flagged Table"),
    ]

    REQUIRED_COLUMNS = [
        "id_string",
        "no_of_violations_left",
        "no_of_violations_right",
        "no_of_total_violations",
        "fence_low",
        "fence_high",
        "quantile_values",
        "q_list",
    ]

    MAX_FLAGGED_POINTS = 1000

    @dataclass
    class PickleLoaded(Message):
        path: Path
        df: pd.DataFrame

    @dataclass
    class FlaggedPointsLoaded(Message):
        row_idx: int
        title: str
        flagged_rows: list[tuple[str, str, str, str]]

    @dataclass
    class FlaggedPointsFailed(Message):
        row_idx: int
        message: str

    def __init__(
        self,
        root: Path,
        model: str,
        date_str: str,
    ) -> None:
        super().__init__()
        self.root = root.expanduser().resolve()
        print(self.root)
        self.model = model
        self.date_str = date_str
        self.data_yaml_file = DATA_YAML_FILE.expanduser().resolve()
        self.strata_file = STRATA_FILE.expanduser().resolve()

        self.files: list[Path] = []
        self.df: pd.DataFrame | None = None
        self.view_df: pd.DataFrame | None = None
        self.current_path: Path | None = None

    def compose(self) -> ComposeResult:
        yield Header()

        with Vertical(id="left"):
            yield Static(f"Root:\n{self.root}", id="root_label")
            yield ListView(id="file_list")

        with Vertical(id="right"):
            yield Static("Select a .pkl file", id="file_summary")
            yield DataTable(id="preview")
            yield Static("Select a row to summarize and plot", id="row_summary")
            with Horizontal(id="plot_area"):
                with Vertical(id="flagged_panel"):
                    yield Static("All flagged data points", id="flagged_title")
                    yield DataTable(id="flagged_table")
                yield QuantilePlot(id="plot_right")

        yield Footer()

    def on_mount(self) -> None:
        preview = self.query_one("#preview", DataTable)
        preview.cursor_type = "row"
        preview.zebra_stripes = True

        flagged = self.query_one("#flagged_table", DataTable)
        flagged.cursor_type = "row"
        flagged.zebra_stripes = True
        flagged.add_columns("lat", "lon", "time", "data")

        self.reload_file_list()

        self.query_one("#flagged_title", Static).update("All flagged data points")
        self.clear_flagged_table("Select a file")
        self.query_one("#plot_right", QuantilePlot).show_message("Select a file")

    def action_reload_files(self) -> None:
        self.reload_file_list()

    def action_focus_files(self) -> None:
        self.query_one("#file_list", ListView).focus()

    def action_focus_table(self) -> None:
        self.query_one("#preview", DataTable).focus()

    def action_focus_flagged_table(self) -> None:
        self.query_one("#flagged_table", DataTable).focus()

    def action_activate_current(self) -> None:
        focused = self.focused

        if focused is None:
            return

        if focused.id == "file_list":
            lv = self.query_one("#file_list", ListView)
            if lv.index is not None and 0 <= lv.index < len(self.files):
                self.load_pickle(self.files[lv.index])
            return

        if focused.id == "preview":
            table = self.query_one("#preview", DataTable)
            row_idx = table.cursor_row
            if row_idx is not None and row_idx >= 0:
                self.update_row_summary(row_idx)
                self.plot_row(row_idx)

    def reload_file_list(self) -> None:
        self.files = sorted(self.root.rglob("*.pkl"))

        lv = self.query_one("#file_list", ListView)
        lv.clear()

        for i, path in enumerate(self.files):
            label = parse_file_label(path, self.root)
            lv.append(ListItem(Label(label), id=f"file_{i}"))

        if self.files:
            lv.index = 0

        self.query_one("#file_summary", Static).update(
            f"Root: {self.root}\n"
            f"Found {len(self.files)} pickle files\n"
            f"Model: {self.model}\n"
            f"Date: {self.date_str}\n\n"
            f"Use arrows + Enter on a file.\n"
            f"Press 'f' for files, 't' for preview table, 'g' for flagged table."
        )

    @on(ListView.Selected, "#file_list")
    def handle_file_selected(self, event: ListView.Selected) -> None:
        if event.item is None or event.item.id is None:
            return

        idx = int(event.item.id.split("_")[-1])
        self.load_pickle(self.files[idx])

    @work(thread=True, exclusive=True)
    def load_pickle(self, path: Path) -> None:
        try:
            obj = pd.read_pickle(path)

            if not isinstance(obj, pd.DataFrame):
                self.notify(f"{path.name} is not a pandas DataFrame", severity="error")
                return

            missing = [c for c in self.REQUIRED_COLUMNS if c not in obj.columns]
            if missing:
                self.notify(
                    f"{path.name} missing required columns: {missing}",
                    severity="error",
                )
                return

            self.post_message(self.PickleLoaded(path=path, df=obj))

        except Exception as e:
            self.notify(f"Failed to load {path.name}: {e}", severity="error")

    @on(PickleLoaded)
    def handle_pickle_loaded(self, event: PickleLoaded) -> None:
        self.current_path = event.path
        self.df = event.df
        self.view_df = None

        self.update_file_summary()
        self.update_preview()
        self.update_row_summary(None)

        table = self.query_one("#preview", DataTable)
        if self.view_df is not None and len(self.view_df) > 0:
            table.focus()
            table.move_cursor(row=0)
            self.update_row_summary(0)
            self.plot_row(0)
        else:
            self.clear_flagged_table("No rows with nonzero total violations")
            self.query_one("#plot_right", QuantilePlot).show_message("No data to plot")

    def update_file_summary(self) -> None:
        assert self.df is not None
        assert self.current_path is not None

        match = FILE_RE.match(self.current_path.name)
        if match:
            model = match.group("model").upper()
            date = match.group("date").replace("_", "-")
            ref = match.group("ref")
        else:
            model = "UNKNOWN"
            date = "UNKNOWN"
            ref = "UNKNOWN"

        total_left = int(self.df["no_of_violations_left"].sum())
        total_right = int(self.df["no_of_violations_right"].sum())
        total_total = int(self.df["no_of_total_violations"].sum())
        nonzero_rows = int((self.df["no_of_total_violations"] != 0).sum())

        text = (
            f"Model: {model}, Date: {date}, Historical Reference Date: {ref}\n"
            f"File: {self.current_path.relative_to(self.root)}\n"
            f"Total left violations: {total_left:,}\n"
            f"Total right violations: {total_right:,}\n"
            f"Total violations: {total_total:,} across {nonzero_rows:,} data slices\n"
            f"\nPress 'f' for files, 't' for preview table, 'g' for flagged table."
        )
        self.query_one("#file_summary", Static).update(text)

    def update_preview(self) -> None:
        assert self.df is not None

        table = self.query_one("#preview", DataTable)
        table.clear(columns=True)

        table.add_columns(
            "id_string (Collection|Variable|Level|Latitude Stratum)",
            "no_of_total_violations",
            "no_of_violations_left",
            "no_of_violations_right",
        )

        self.view_df = self.df[
            self.df["no_of_total_violations"] != 0
        ].sort_values(by="no_of_total_violations", ascending=False).copy()

        for _, row in self.view_df.iterrows():
            table.add_row(
                shorten(row["id_string"], 90),
                str(row["no_of_total_violations"]),
                str(row["no_of_violations_left"]),
                str(row["no_of_violations_right"]),
            )

    @on(DataTable.RowHighlighted, "#preview")
    def handle_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        row_idx = event.cursor_row
        self.update_row_summary(row_idx)
        if row_idx is not None and row_idx >= 0:
            self.plot_row(row_idx)

    def update_row_summary(self, row_idx: int | None) -> None:
        if self.view_df is None or row_idx is None or row_idx < 0 or row_idx >= len(self.view_df):
            self.query_one("#row_summary", Static).update(
                "Select a row to summarize and plot"
            )
            return

        row = self.view_df.iloc[row_idx]

        text = (
            f"Visible row: {row_idx}\n"
            f"id_string: {row['id_string']} (Collection|Variable|Level|Latitude Stratum)\n"
            f"no_of_total_violations: {row['no_of_total_violations']}\n"
            f"no_of_violations_left: {row['no_of_violations_left']}\n"
            f"no_of_violations_right: {row['no_of_violations_right']}\n"
            f"fence_low: {row['fence_low']}\n"
            f"fence_high: {row['fence_high']}\n"
        )
        self.query_one("#row_summary", Static).update(text)

    def clear_flagged_table(self, message: str) -> None:
        table = self.query_one("#flagged_table", DataTable)
        table.clear(columns=True)
        table.add_columns("lat", "lon", "time", "data")
        self.query_one("#flagged_title", Static).update(f"All flagged data points\n{message}")
        table.refresh()

    def populate_flagged_table(self, title: str, rows: list[tuple[str, str, str, str]]) -> None:
        table = self.query_one("#flagged_table", DataTable)
        table.clear(columns=True)
        table.add_columns("lat", "lon", "time", "data")

        for lat, lon, time_val, data_val in rows:
            table.add_row(lat, lon, time_val, data_val)

        self.query_one("#flagged_title", Static).update(title)

        if rows:
            table.move_cursor(row=0)
        table.refresh()

    def plot_row(self, row_idx: int) -> None:
        if self.view_df is None or row_idx < 0 or row_idx >= len(self.view_df):
            return

        row = self.view_df.iloc[row_idx]

        x_vals = to_1d_float_list(row["q_list"])
        y_vals = to_1d_float_list(row["quantile_values"])

        fence_low = None
        fence_high = None

        try:
            fence_low = float(row["fence_low"])
        except Exception:
            pass

        try:
            fence_high = float(row["fence_high"])
        except Exception:
            pass

        id_string = str(row["id_string"])

        self.clear_flagged_table("Loading flagged data...")

        self.query_one("#plot_right", QuantilePlot).show_row_plot(
            x_vals=x_vals,
            y_vals=y_vals,
            title="Historical reference quantiles",
            fence_low=fence_low,
            fence_high=fence_high,
        )

        self.load_flagged_points(
            row_idx=row_idx,
            id_string=id_string,
            fence_low=fence_low,
            fence_high=fence_high,
        )

    @work(thread=True, exclusive=True)
    def load_flagged_points(
        self,
        row_idx: int,
        id_string: str,
        fence_low: float | None,
        fence_high: float | None,
    ) -> None:
        try:
            coll, var, lev_str, sname = id_string.split("|", 3)
            lev_val = None if lev_str == "None" else lev_str

            dt = datetime.strptime(self.date_str, "%Y-%m-%d")
            _, collection_dict, excluded = list_files_and_excluded_vars(
                model=self.model.upper(),
                date=dt,
                data_yaml_file=self.data_yaml_file,
            )
            strata = load_strata(self.strata_file)

            ds = xr.open_mfdataset(
                collection_dict[coll],
                combine="by_coords",
                drop_variables=excluded,
                data_vars="minimal",
                coords="minimal",
                compat="no_conflicts",
                engine="h5netcdf",
                chunks="auto",
                parallel=True,
            )

            lat = pick_name(ds, LAT_NAMES)
            lon = pick_name(ds, LON_NAMES)
            lev = pick_name(ds, LEV_NAMES)
            time = pick_name(ds, TIME_NAMES)

            if lat is None or lon is None:
                raise ValueError("Could not identify latitude/longitude coordinate names.")
            if var not in ds:
                raise ValueError(f"Variable {var!r} not found in dataset.")

            da = ds[var]

            if lev_val is not None and lev is not None and lev in da.coords:
                try:
                    da = da.sel({lev: float(lev_val)})
                except Exception:
                    da = da.sel({lev: lev_val})

            lat_min = strata[sname]["lat"]["min"]
            lat_max = strata[sname]["lat"]["max"]
            lat_vals_full = ds[lat].values

            if np.all(np.diff(lat_vals_full) > 0):
                da = da.sel({lat: slice(lat_min, lat_max)})
            else:
                da = da.sel({lat: slice(lat_max, lat_min)})

            if time is not None and time in da.dims:
                da = da.transpose(time, lat, lon).compute()
                data = np.asarray(da.values, dtype=float)
                time_vals = np.asarray(da[time].values)
            else:
                da = da.transpose(lat, lon).compute()
                data_2d = np.asarray(da.values, dtype=float)
                data = data_2d[np.newaxis, :, :]
                time_vals = np.asarray(["NA"], dtype=object)

            lat_vals = np.asarray(da[lat].values, dtype=float)
            lon_vals = np.asarray(da[lon].values, dtype=float)

            flagged_mask = np.zeros_like(data, dtype=bool)
            if fence_low is not None:
                flagged_mask |= np.isfinite(data) & (data < fence_low)
            if fence_high is not None:
                flagged_mask |= np.isfinite(data) & (data > fence_high)

            flagged_indices = np.argwhere(flagged_mask)
            total_flagged = len(flagged_indices)
            flagged_indices = flagged_indices[: self.MAX_FLAGGED_POINTS]

            flagged_rows: list[tuple[str, str, str, str]] = []
            for t_idx, lat_idx, lon_idx in flagged_indices:
                lat_value = lat_vals[lat_idx]
                lon_value = lon_vals[lon_idx]
                time_value = time_vals[t_idx] if t_idx < len(time_vals) else "NA"
                data_value = data[t_idx, lat_idx, lon_idx]

                flagged_rows.append(
                    (
                        str(lat_value),
                        str(lon_value),
                        str(time_value),
                        str(data_value),
                    )
                )

            shown = len(flagged_rows)
            if total_flagged > shown:
                title = f"All flagged data points (showing first {shown:,} of {total_flagged:,})"
            else:
                title = f"All flagged data points ({shown:,})"

            self.post_message(
                self.FlaggedPointsLoaded(
                    row_idx=row_idx,
                    title=title,
                    flagged_rows=flagged_rows,
                )
            )

        except Exception:
            tb = traceback.format_exc()
            self.post_message(
                self.FlaggedPointsFailed(
                    row_idx=row_idx,
                    message=(
                        f"Failed to load external data for {id_string}\n"
                        f"row_idx={row_idx}\n"
                        f"model={self.model}\n"
                        f"date_str={self.date_str}\n"
                        f"root={self.root}\n\n"
                        f"Traceback:\n{tb}"
                    ),
                )
            )

    @on(FlaggedPointsLoaded)
    def handle_flagged_points_loaded(self, event: FlaggedPointsLoaded) -> None:
        preview = self.query_one("#preview", DataTable)
        current_row = preview.cursor_row

        if current_row is None or current_row != event.row_idx:
            return

        self.populate_flagged_table(event.title, event.flagged_rows)

    @on(FlaggedPointsFailed)
    def handle_flagged_points_failed(self, event: FlaggedPointsFailed) -> None:
        preview = self.query_one("#preview", DataTable)
        current_row = preview.cursor_row

        if current_row is None or current_row != event.row_idx:
            return

        self.clear_flagged_table("Failed to load flagged data")
        self.notify(event.message, severity="error")


if __name__ == "__main__":
    base_root = Path(
        sys.argv[1] if len(sys.argv) > 1 else "~/JupyterLinks/nobackup/quads_results"
    ).expanduser()

    model = input("Enter model name (e.g. geosfp): ").strip()
    date_str = input("Enter date (YYYY-MM-DD, e.g. 2024-02-01): ").strip()

    try:
        year, month, day = date_str.split("-")
    except ValueError:
        print("Date must be in YYYY-MM-DD format.")
        raise SystemExit(1)

    root = base_root / model.upper() / year / month / day
    print("root:", root)
    app = PklBrowser(
        root=root,
        model=model,
        date_str=date_str,
    )
    app.run()
