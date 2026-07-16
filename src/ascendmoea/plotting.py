"""Optional Plotly figures for populations, fronts, and metric histories."""

from __future__ import annotations

import plotly.graph_objects as go
import torch

from .core import to_tensor


def _plot_tensor(x) -> torch.Tensor:
    """Convert plotting input to a detached float64 CPU tensor."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().to(torch.float64)
    return to_tensor(x, device=torch.device("cpu"), dtype=torch.float64)


def _as_pf_grid(pf):
    """Return a validated three-matrix Pareto-front grid or ``None``."""
    if not isinstance(pf, (tuple, list)) or len(pf) != 3:
        return None
    try:
        x = _plot_tensor(pf[0])
        y = _plot_tensor(pf[1])
        z = _plot_tensor(pf[2])
    except Exception:
        return None
    if x.ndim != 2 or y.ndim != 2 or z.ndim != 2:
        return None
    if x.shape != y.shape or y.shape != z.shape:
        return None
    return x, y, z


def _base_layout(fig: go.Figure, title: str = ""):
    """Apply the shared publication-oriented figure layout."""
    fig.update_layout(
        title=title,
        template="plotly_white",
        font=dict(family="Times New Roman", size=14, color="#1c2d4d"),
        margin=dict(l=24, r=16, t=44, b=24),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#fbfcff",
        legend=dict(
            bgcolor="rgba(255,255,255,0.75)", bordercolor="#d7deed", borderwidth=1, x=1.0, y=1.0
        ),
        height=600,
        transition=dict(duration=0),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e7ecf7", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#e7ecf7", zeroline=False)
    return fig


def draw_obj(obj, pf=None, title: str = "Population (objectives)") -> go.Figure:
    """Draw a two- or three-objective population and optional reference front."""
    obj = _plot_tensor(obj)
    if obj.ndim != 2 or obj.shape[1] not in {2, 3}:
        raise ValueError("obj must have shape (N, 2) or (N, 3)")
    if obj.shape[1] == 2:
        fig = go.Figure()
        if pf is not None:
            pf_t = _plot_tensor(pf)
            if pf_t.numel() and pf_t.shape[1] == 2:
                fig.add_trace(
                    go.Scatter(
                        x=pf_t[:, 0].tolist(),
                        y=pf_t[:, 1].tolist(),
                        mode="lines",
                        line=dict(color="#111", width=2),
                        name="PF",
                    )
                )
        fig.add_trace(
            go.Scatter(
                x=obj[:, 0].tolist(),
                y=obj[:, 1].tolist(),
                mode="markers",
                marker=dict(size=7, color="#b8b8b8", line=dict(color="#555", width=1)),
                name="Population",
            )
        )
        fig.update_xaxes(
            title_text="<i>f</i><sub>1</sub>",
            showline=True,
            linewidth=1,
            linecolor="#111",
            mirror=True,
        )
        fig.update_yaxes(
            title_text="<i>f</i><sub>2</sub>",
            showline=True,
            linewidth=1,
            linecolor="#111",
            mirror=True,
        )
        return _base_layout(fig, title)
    fig = go.Figure()
    if pf is not None:
        pf_t = _plot_tensor(pf)
        if pf_t.numel() and pf_t.shape[1] == 3:
            fig.add_trace(
                go.Scatter3d(
                    x=pf_t[:, 0].tolist(),
                    y=pf_t[:, 1].tolist(),
                    z=pf_t[:, 2].tolist(),
                    mode="markers",
                    marker=dict(size=2, color="#999"),
                    name="PF",
                )
            )
    fig.add_trace(
        go.Scatter3d(
            x=obj[:, 0].tolist(),
            y=obj[:, 1].tolist(),
            z=obj[:, 2].tolist(),
            mode="markers",
            marker=dict(size=4, color="#b8b8b8", line=dict(color="#555", width=1)),
            name="Population",
        )
    )
    fig.update_layout(
        scene=dict(
            xaxis_title="<i>f</i><sub>1</sub>",
            yaxis_title="<i>f</i><sub>2</sub>",
            zaxis_title="<i>f</i><sub>3</sub>",
        )
    )
    return _base_layout(fig, title)


def draw_problem_obj(
    problem, pop_obj, title: str = "Population (objectives)", include_pf: bool = True
) -> go.Figure:
    """Draw a population with reference geometry obtained from its problem."""
    pop_obj = _plot_tensor(pop_obj)
    if pop_obj.ndim == 1 and pop_obj.numel() == 0:
        pop_obj = torch.empty((0, problem.M), dtype=torch.float64)
    if pop_obj.ndim != 2 or pop_obj.shape[1] not in {2, 3}:
        raise ValueError("pop_obj must have shape (N, 2) or (N, 3)")
    pf = problem.pf
    opt = problem.optimum
    if pop_obj.shape[1] == 2:
        fig = go.Figure()
        if include_pf and pf is not None:
            pf_grid = _as_pf_grid(pf)
            if pf_grid is not None:
                xg, yg, zg = pf_grid
                fig.add_trace(
                    go.Heatmap(
                        x=xg[0, :].tolist(),
                        y=yg[:, 0].tolist(),
                        z=zg.tolist(),
                        showscale=False,
                        colorscale=[[0.0, "#d6deec"], [1.0, "#d6deec"]],
                        opacity=0.55,
                        name="Feasible PF Region",
                    )
                )
            else:
                try:
                    pf_t = _plot_tensor(pf)
                    if pf_t.numel() and pf_t.shape[1] == 2:
                        fig.add_trace(
                            go.Scatter(
                                x=pf_t[:, 0].tolist(),
                                y=pf_t[:, 1].tolist(),
                                mode="lines",
                                line=dict(color="#111", width=1),
                                name="PF",
                            )
                        )
                except Exception:
                    pass
        if include_pf and opt is not None:
            try:
                opt_t = _plot_tensor(opt)
                if opt_t.numel() and opt_t.shape[1] == 2 and opt_t.shape[0] > 1:
                    fig.add_trace(
                        go.Scatter(
                            x=opt_t[:, 0].tolist(),
                            y=opt_t[:, 1].tolist(),
                            mode="markers",
                            marker=dict(size=4, color="#111"),
                            name="True PF",
                        )
                    )
            except Exception:
                pass
        fig.add_trace(
            go.Scattergl(
                x=pop_obj[:, 0].tolist(),
                y=pop_obj[:, 1].tolist(),
                mode="markers",
                marker=dict(size=7, color="#b8b8b8", line=dict(color="#555", width=1)),
                name="Population",
            )
        )
        fig.update_xaxes(
            title_text="<i>f</i><sub>1</sub>",
            showline=True,
            linewidth=1,
            linecolor="#111",
            mirror=True,
        )
        fig.update_yaxes(
            title_text="<i>f</i><sub>2</sub>",
            showline=True,
            linewidth=1,
            linecolor="#111",
            mirror=True,
        )
        return _base_layout(fig, title)
    fig = go.Figure()
    if include_pf and pf is not None:
        pf_grid = _as_pf_grid(pf)
        if pf_grid is not None:
            xg, yg, zg = pf_grid
            fig.add_trace(
                go.Surface(
                    x=xg.tolist(),
                    y=yg.tolist(),
                    z=zg.tolist(),
                    showscale=False,
                    opacity=0.75,
                    colorscale=[[0.0, "#8fa8d4"], [1.0, "#8fa8d4"]],
                    name="PF",
                )
            )
        else:
            try:
                pf_t = _plot_tensor(pf)
                if pf_t.numel() and pf_t.shape[1] == 3:
                    fig.add_trace(
                        go.Scatter3d(
                            x=pf_t[:, 0].tolist(),
                            y=pf_t[:, 1].tolist(),
                            z=pf_t[:, 2].tolist(),
                            mode="markers",
                            marker=dict(size=2, color="#999"),
                            name="PF",
                        )
                    )
            except Exception:
                pass
    fig.add_trace(
        go.Scatter3d(
            x=pop_obj[:, 0].tolist(),
            y=pop_obj[:, 1].tolist(),
            z=pop_obj[:, 2].tolist(),
            mode="markers",
            marker=dict(size=4, color="#b8b8b8", line=dict(color="#555", width=1)),
            name="Population",
        )
    )
    fig.update_layout(
        scene=dict(
            xaxis_title="<i>f</i><sub>1</sub>",
            yaxis_title="<i>f</i><sub>2</sub>",
            zaxis_title="<i>f</i><sub>3</sub>",
        )
    )
    return _base_layout(fig, title)


def update_population_trace(fig: go.Figure, pop_obj) -> go.Figure:
    """Replace the population trace of an existing figure in place."""
    pop_obj = _plot_tensor(pop_obj)
    if pop_obj.ndim == 1 and pop_obj.numel() == 0:
        return fig
    for tr in fig.data:
        if getattr(tr, "name", "") == "Population":
            if pop_obj.shape[1] == 2 and hasattr(tr, "x"):
                tr.x = pop_obj[:, 0].tolist()
                tr.y = pop_obj[:, 1].tolist()
            elif pop_obj.shape[1] >= 3 and hasattr(tr, "z"):
                tr.x = pop_obj[:, 0].tolist()
                tr.y = pop_obj[:, 1].tolist()
                tr.z = pop_obj[:, 2].tolist()
            break
    return fig


def draw_dec(dec, title: str = "Population (variables)") -> go.Figure:
    """Draw decision vectors as points or parallel dimension traces."""
    dec = _plot_tensor(dec)
    if dec.ndim != 2 or dec.shape[1] == 0:
        raise ValueError("dec must have shape (N, D) with D >= 1")
    if dec.shape[1] == 1:
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=torch.arange(1, dec.shape[0] + 1).tolist(),
                    y=dec[:, 0].tolist(),
                    mode="markers",
                    marker=dict(size=7, color="#b8b8b8", line=dict(color="#555", width=1)),
                    name="Population",
                )
            ]
        )
        fig.update_xaxes(title_text="Solution index")
        fig.update_yaxes(title_text="<i>x</i><sub>1</sub>")
        return _base_layout(fig, title)
    if dec.shape[1] == 2:
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=dec[:, 0].tolist(),
                    y=dec[:, 1].tolist(),
                    mode="markers",
                    marker=dict(size=7, color="#b8b8b8", line=dict(color="#555", width=1)),
                    name="Population",
                )
            ]
        )
        fig.update_xaxes(title_text="<i>x</i><sub>1</sub>")
        fig.update_yaxes(title_text="<i>x</i><sub>2</sub>")
        return _base_layout(fig, title)
    if dec.shape[1] == 3:
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=dec[:, 0].tolist(),
                    y=dec[:, 1].tolist(),
                    z=dec[:, 2].tolist(),
                    mode="markers",
                    marker=dict(size=4, color="#b8b8b8", line=dict(color="#555", width=1)),
                    name="Population",
                )
            ]
        )
        fig.update_layout(
            scene=dict(
                xaxis_title="<i>x</i><sub>1</sub>",
                yaxis_title="<i>x</i><sub>2</sub>",
                zaxis_title="<i>x</i><sub>3</sub>",
            )
        )
        return _base_layout(fig, title)
    x = torch.arange(1, dec.shape[1] + 1, dtype=torch.float64)
    fig = go.Figure()
    for i in range(dec.shape[0]):
        fig.add_trace(
            go.Scatter(
                x=x.tolist(),
                y=dec[i].tolist(),
                mode="lines",
                line=dict(color="rgba(90,90,90,0.25)", width=1),
                showlegend=False,
            )
        )
    fig.update_xaxes(
        title_text="Dimension No.", showline=True, linewidth=1, linecolor="#111", mirror=True
    )
    fig.update_yaxes(title_text="Value", showline=True, linewidth=1, linecolor="#111", mirror=True)
    return _base_layout(fig, title)


def draw_metric_curve(fes, values, name: str) -> go.Figure:
    """Draw an indicator value against cumulative function evaluations."""
    fes = _plot_tensor(fes).reshape(-1)
    values = _plot_tensor(values).reshape(-1)
    if fes.numel() != values.numel():
        raise ValueError("fes and values must contain the same number of elements")
    fig = go.Figure(
        data=[
            go.Scatter(
                x=fes.tolist(),
                y=values.tolist(),
                mode="lines+markers",
                line=dict(color="#111", width=2),
                marker=dict(size=6, color="#666"),
                name=name,
            )
        ]
    )
    fig.update_xaxes(
        title_text="Number of function evaluations",
        showline=True,
        linewidth=1,
        linecolor="#111",
        mirror=True,
    )
    fig.update_yaxes(title_text=name, showline=True, linewidth=1, linecolor="#111", mirror=True)
    return _base_layout(fig, name)


__all__ = [
    "draw_dec",
    "draw_metric_curve",
    "draw_obj",
    "draw_problem_obj",
    "update_population_trace",
]
