"""
Аудит качества данных и EDA
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .common import get_project_root, load_config, ensure_dir, format_number
from .ingest import load_bronze_data


def run_audit(
    df: Optional[pd.DataFrame] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Провести аудит качества данных

    Returns:
        Словарь с результатами аудита
    """
    if config is None:
        config = load_config()

    if df is None:
        df = load_bronze_data(config)

    print("="*60)
    print("АУДИТ КАЧЕСТВА ДАННЫХ")
    print("="*60)

    audit_results = {
        "timestamp": datetime.now().isoformat(),
        "basic_stats": {},
        "missing_values": {},
        "infinities": {},
        "duplicates": {},
        "data_types": {},
        "target_distribution": {},
        "issues": []
    }

    # 1. Базовая статистика
    print("\n📊 Базовая статистика:")
    audit_results["basic_stats"] = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "memory_mb": round(df.memory_usage(deep=True).sum() / (1024*1024), 2)
    }
    print(f"   Строк: {format_number(len(df))}")
    print(f"   Колонок: {len(df.columns)}")
    print(f"   Память: {audit_results['basic_stats']['memory_mb']:.1f} MB")

    # 2. Типы данных
    print("\n📊 Типы данных:")
    dtype_counts = df.dtypes.astype(str).value_counts().to_dict()
    audit_results["data_types"] = dtype_counts
    for dtype, count in dtype_counts.items():
        print(f"   {dtype}: {count}")

    # 3. Пропуски (NaN)
    print("\n📊 Пропуски (NaN):")
    nan_counts = df.isna().sum()
    nan_cols = nan_counts[nan_counts > 0].sort_values(ascending=False)

    audit_results["missing_values"]["total_cells_with_nan"] = int(nan_counts.sum())
    audit_results["missing_values"]["columns_with_nan"] = len(nan_cols)
    audit_results["missing_values"]["details"] = {
        col: {"count": int(count), "percent": round(100*count/len(df), 4)}
        for col, count in nan_cols.items()
    }

    if len(nan_cols) > 0:
        print(f"   Колонок с NaN: {len(nan_cols)}")
        for col, count in nan_cols.head(10).items():
            pct = 100 * count / len(df)
            print(f"   - {col}: {format_number(count)} ({pct:.2f}%)")
        audit_results["issues"].append(f"Found {len(nan_cols)} columns with NaN values")
    else:
        print("   ✅ Пропусков нет")

    # 4. Бесконечности
    print("\n📊 Бесконечности (Inf):")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = {}

    for col in numeric_cols:
        pos_inf = (df[col] == np.inf).sum()
        neg_inf = (df[col] == -np.inf).sum()
        total_inf = pos_inf + neg_inf
        if total_inf > 0:
            inf_counts[col] = {
                "positive_inf": int(pos_inf),
                "negative_inf": int(neg_inf),
                "total": int(total_inf),
                "percent": round(100*total_inf/len(df), 4)
            }

    audit_results["infinities"]["columns_with_inf"] = len(inf_counts)
    audit_results["infinities"]["details"] = inf_counts

    if inf_counts:
        print(f"   Колонок с Inf: {len(inf_counts)}")
        for col, info in inf_counts.items():
            print(f"   - {col}: {format_number(info['total'])} ({info['percent']:.2f}%)")
        audit_results["issues"].append(f"Found {len(inf_counts)} columns with Inf values")
    else:
        print("   ✅ Бесконечностей нет")

    # 5. Дубликаты
    print("\n📊 Дубликаты:")
    full_dups = df.duplicated().sum()
    audit_results["duplicates"]["full_duplicates"] = {
        "count": int(full_dups),
        "percent": round(100*full_dups/len(df), 2)
    }
    print(f"   Полных дубликатов: {format_number(full_dups)} ({100*full_dups/len(df):.2f}%)")

    analysis_cols = [c for c in df.columns if not c.startswith('_')]
    dups_no_meta = df.duplicated(subset=analysis_cols).sum()
    audit_results["duplicates"]["without_meta"] = {
        "count": int(dups_no_meta),
        "percent": round(100*dups_no_meta/len(df), 2)
    }
    print(f"   Без мета-колонок: {format_number(dups_no_meta)} ({100*dups_no_meta/len(df):.2f}%)")

    if full_dups > 0:
        audit_results["issues"].append(f"Found {full_dups:,} duplicate rows ({100*full_dups/len(df):.1f}%)")

    # 6. Проверка целевой переменной
    target_col = config["ingestion"]["target_column"]
    if target_col in df.columns:
        print(f"\n📊 Целевая переменная ({target_col}):")
        target_counts = df[target_col].value_counts()
        audit_results["target_distribution"] = {
            label: {"count": int(count), "percent": round(100*count/len(df), 2)}
            for label, count in target_counts.items()
        }

        for label, count in target_counts.items():
            pct = 100 * count / len(df)
            print(f"   {label}: {format_number(count)} ({pct:.2f}%)")

    # 7. Проверка дублирующихся колонок
    print("\n📊 Проверка колонок:")
    col_names = df.columns.tolist()
    duplicate_cols = [col for col in set(col_names) if col_names.count(col) > 1]

    if duplicate_cols:
        print(f"   ⚠️ Дублирующиеся названия колонок: {duplicate_cols}")
        audit_results["issues"].append(f"Duplicate column names: {duplicate_cols}")
    else:
        fwd_header_cols = [c for c in col_names if 'Fwd Header Length' in c]
        if len(fwd_header_cols) > 1:
            print(f"   ⚠️ Похожие колонки: {fwd_header_cols}")
            audit_results["issues"].append(f"Similar column names found: {fwd_header_cols}")
        else:
            print("   ✅ Колонки уникальны")

    # 8. Проверка диапазонов
    print("\n📊 Экстремальные значения:")
    extreme_cols = []
    for col in list(numeric_cols)[:20]:
        valid_data = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid_data) > 0:
            q1, q99 = valid_data.quantile([0.01, 0.99])
            min_val, max_val = valid_data.min(), valid_data.max()
            if max_val > q99 * 100 or (q1 != 0 and min_val < q1 * 100):
                extreme_cols.append(col)

    if extreme_cols:
        print(f"   Колонок с экстремальными выбросами: {len(extreme_cols)}")
        audit_results["issues"].append(f"Columns with extreme outliers: {extreme_cols[:5]}...")

    print("\n" + "="*60)
    print(f"ИТОГО ПРОБЛЕМ: {len(audit_results['issues'])}")
    for issue in audit_results["issues"]:
        print(f"   ⚠️ {issue}")
    print("="*60)

    return audit_results


def run_eda(
    df: Optional[pd.DataFrame] = None,
    config: Optional[Dict[str, Any]] = None,
    output_path: Optional[Path] = None,
    save_format: str = "png"  # "png" или "html"
) -> Dict[str, Path]:
    """
    Провести EDA с визуализациями

    Args:
        df: DataFrame
        config: Конфигурация
        output_path: Путь для сохранения
        save_format: Формат сохранения ("png" или "html")

    Returns:
        Словарь с путями к созданным визуализациям
    """
    if config is None:
        config = load_config()

    if df is None:
        df = load_bronze_data(config)

    root = get_project_root()
    if output_path is None:
        output_path = root / config["paths"]["reports"]

    ensure_dir(output_path)
    ensure_dir(output_path / "figures")

    created_files = {}
    eda_config = config.get("eda", {})
    sample_size = eda_config.get("sample_size", 100000)

    # Сэмплируем для тяжёлых визуализаций
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
        print(f"📊 Using sample of {sample_size:,} rows for visualizations")
    else:
        df_sample = df

    target_col = config["ingestion"]["target_column"]

    def save_figure(fig: go.Figure, name: str) -> Path:
        """Сохранить фигуру в нужном формате"""
        if save_format == "png":
            path = output_path / "figures" / f"{name}.png"
            fig.write_image(str(path), width=1200, height=800, scale=2)
        else:
            path = output_path / "figures" / f"{name}.html"
            fig.write_html(str(path))
        return path

    # 1. Распределение классов
    print("\n📈 Creating: Class Distribution...")
    fig = create_class_distribution_plot(df, target_col)
    path = save_figure(fig, "01_class_distribution")
    created_files["class_distribution"] = path

    # 2. Распределение по дням
    if "_day" in df.columns:
        print("📈 Creating: Distribution by Day...")
        fig = create_day_distribution_plot(df, target_col)
        path = save_figure(fig, "02_day_distribution")
        created_files["day_distribution"] = path

    # 3. Дисбаланс классов (log scale)
    print("📈 Creating: Class Imbalance...")
    fig = create_class_imbalance_plot(df, target_col)
    path = save_figure(fig, "03_class_imbalance_log")
    created_files["class_imbalance"] = path

    # 4. Корреляционная матрица
    print("📈 Creating: Correlation Matrix...")
    fig = create_correlation_matrix(df_sample, config)
    path = save_figure(fig, "04_correlation_matrix")
    created_files["correlation_matrix"] = path

    # 5. Распределение числовых признаков
    print("📈 Creating: Feature Distributions...")
    fig = create_feature_distributions(df_sample, config)
    path = save_figure(fig, "05_feature_distributions")
    created_files["feature_distributions"] = path

    # 6. Boxplots для выбросов
    print("📈 Creating: Outlier Analysis...")
    fig = create_outlier_boxplots(df_sample, config)
    path = save_figure(fig, "06_outlier_boxplots")
    created_files["outlier_boxplots"] = path

    # 7. Признаки по классам
    print("📈 Creating: Features by Class...")
    fig = create_features_by_class(df_sample, target_col, config)
    path = save_figure(fig, "07_features_by_class")
    created_files["features_by_class"] = path

    # 8. Топ признаков для бинарной классификации
    print("📈 Creating: Top Features Analysis...")
    fig = create_top_features_analysis(df_sample, target_col)
    path = save_figure(fig, "08_top_features_binary")
    created_files["top_features"] = path

    print(f"\n✅ Created {len(created_files)} visualizations in {output_path / 'figures'}")

    return created_files


def create_class_distribution_plot(df: pd.DataFrame, target_col: str) -> go.Figure:
    """Создать график распределения классов"""
    class_counts = df[target_col].value_counts().reset_index()
    class_counts.columns = ['Class', 'Count']
    class_counts['Percentage'] = (class_counts['Count'] / len(df) * 100).round(2)

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "bar"}, {"type": "pie"}]],
        subplot_titles=("Количество по классам", "Доля классов"),
        horizontal_spacing=0.15
    )

    colors = ['#2ecc71' if c == 'BENIGN' else '#e74c3c' for c in class_counts['Class']]

    fig.add_trace(
        go.Bar(
            x=class_counts['Class'],
            y=class_counts['Count'],
            text=[f'{c:,}' for c in class_counts['Count']],
            textposition='outside',
            marker_color=colors
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Pie(
            labels=class_counts['Class'],
            values=class_counts['Count'],
            textinfo='percent',
            hovertemplate='%{label}: %{value:,}<extra></extra>',
            hole=0.4,
            marker=dict(colors=['#2ecc71' if c == 'BENIGN' else '#e74c3c'
                               for c in class_counts['Class']])
        ),
        row=1, col=2
    )

    fig.update_layout(
        title_text="Распределение классов CIC-IDS-2017",
        title_x=0.5,
        height=500,
        showlegend=False,
        font=dict(size=12)
    )

    fig.update_xaxes(tickangle=45, row=1, col=1)

    return fig


def create_class_imbalance_plot(df: pd.DataFrame, target_col: str) -> go.Figure:
    """Создать график дисбаланса классов (log scale)"""
    class_counts = df[target_col].value_counts().sort_values(ascending=True)

    fig = go.Figure()

    colors = ['#2ecc71' if c == 'BENIGN' else '#e74c3c' for c in class_counts.index]

    fig.add_trace(go.Bar(
        y=class_counts.index,
        x=class_counts.values,
        orientation='h',
        text=[f'{c:,}' for c in class_counts.values],
        textposition='outside',
        marker_color=colors
    ))

    fig.update_layout(
        title='Дисбаланс классов (логарифмическая шкала)',
        title_x=0.5,
        xaxis_title='Количество записей (log scale)',
        yaxis_title='Класс',
        xaxis_type='log',
        height=600,
        margin=dict(l=200)
    )

    return fig


def create_day_distribution_plot(df: pd.DataFrame, target_col: str) -> go.Figure:
    """Создать график распределения по дням"""
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

    day_class = df.groupby(['_day', target_col]).size().reset_index(name='count')

    # Бинарная версия
    day_class['is_attack'] = day_class[target_col].apply(lambda x: 'Attack' if x != 'BENIGN' else 'Benign')
    day_binary = day_class.groupby(['_day', 'is_attack'])['count'].sum().reset_index()

    fig = px.bar(
        day_binary,
        x='_day',
        y='count',
        color='is_attack',
        title='Распределение Benign/Attack по дням недели',
        category_orders={'_day': day_order},
        barmode='group',
        color_discrete_map={'Benign': '#2ecc71', 'Attack': '#e74c3c'}
    )

    fig.update_layout(
        height=500,
        xaxis_title='День недели',
        yaxis_title='Количество записей',
        legend_title='Тип трафика'
    )

    return fig


def create_correlation_matrix(df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
    """Создать корреляционную матрицу"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if not c.startswith('_')]

    max_features = config.get("eda", {}).get("max_features_corr", 30)
    if len(numeric_cols) > max_features:
        variances = df[numeric_cols].var().sort_values(ascending=False)
        numeric_cols = variances.head(max_features).index.tolist()

    df_clean = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    corr_matrix = df_clean.corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        hoverongaps=False,
        colorbar=dict(title='Корреляция')
    ))

    fig.update_layout(
        title='Корреляционная матрица (топ признаков по дисперсии)',
        title_x=0.5,
        height=900,
        width=1000,
        xaxis=dict(tickangle=45)
    )

    return fig


def create_feature_distributions(df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
    """Создать гистограммы распределений признаков"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if not c.startswith('_')]

    top_n = min(16, len(numeric_cols))
    cols_to_plot = numeric_cols[:top_n]

    n_cols = 4
    n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=cols_to_plot
    )

    for i, col in enumerate(cols_to_plot):
        row = i // n_cols + 1
        col_idx = i % n_cols + 1

        valid_data = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        q01, q99 = valid_data.quantile([0.01, 0.99])
        clipped = valid_data.clip(q01, q99)

        fig.add_trace(
            go.Histogram(x=clipped, nbinsx=30, name=col, showlegend=False,
                        marker_color='#3498db'),
            row=row,
            col=col_idx
        )

    fig.update_layout(
        title_text="Распределения признаков (1-99 percentile)",
        title_x=0.5,
        height=200 * n_rows,
        showlegend=False
    )

    return fig


def create_outlier_boxplots(df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
    """Создать boxplots для анализа выбросов"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if not c.startswith('_')]

    top_n = min(12, len(numeric_cols))
    cols_to_plot = numeric_cols[:top_n]

    fig = go.Figure()

    for col in cols_to_plot:
        valid_data = df[col].replace([np.inf, -np.inf], np.nan).dropna()

        if valid_data.std() > 0:
            normalized = (valid_data - valid_data.mean()) / valid_data.std()
        else:
            normalized = valid_data

        fig.add_trace(go.Box(y=normalized.sample(min(10000, len(normalized))),
                             name=col, showlegend=False))

    fig.update_layout(
        title_text="Анализ выбросов (Z-score нормализация)",
        title_x=0.5,
        height=500,
        showlegend=False,
        yaxis_title='Z-score'
    )

    return fig


def create_features_by_class(
    df: pd.DataFrame,
    target_col: str,
    config: Dict[str, Any]
) -> go.Figure:
    """Создать сравнение признаков по классам"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if not c.startswith('_')]

    cols_to_plot = numeric_cols[:6]

    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=cols_to_plot
    )

    df_plot = df.copy()
    df_plot['_binary_class'] = df_plot[target_col].apply(
        lambda x: 'BENIGN' if x == 'BENIGN' else 'ATTACK'
    )

    for i, col in enumerate(cols_to_plot):
        row = i // 3 + 1
        col_idx = i % 3 + 1

        for class_name, color in [('BENIGN', '#2ecc71'), ('ATTACK', '#e74c3c')]:
            class_data = df_plot[df_plot['_binary_class'] == class_name][col]
            class_data = class_data.replace([np.inf, -np.inf], np.nan).dropna()

            if len(class_data) > 10000:
                class_data = class_data.sample(10000)

            q01, q99 = class_data.quantile([0.01, 0.99])
            clipped = class_data.clip(q01, q99)

            fig.add_trace(
                go.Histogram(
                    x=clipped,
                    name=class_name,
                    opacity=0.7,
                    marker_color=color,
                    showlegend=(i == 0),
                    nbinsx=30
                ),
                row=row,
                col=col_idx
            )

    fig.update_layout(
        title_text="Сравнение признаков: BENIGN vs ATTACK",
        title_x=0.5,
        height=600,
        barmode='overlay'
    )

    return fig


def create_top_features_analysis(df: pd.DataFrame, target_col: str) -> go.Figure:
    """Анализ топ признаков для бинарной классификации"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if not c.startswith('_')]

    df_sample = df.sample(min(50000, len(df)), random_state=42).copy()
    df_sample['is_attack'] = (df_sample[target_col] != 'BENIGN').astype(int)

    # Считаем разницу средних для каждого признака
    feature_importance = {}
    for col in numeric_cols:
        valid_data = df_sample[col].replace([np.inf, -np.inf], np.nan)
        if valid_data.isna().sum() < len(valid_data) * 0.5:
            benign_mean = valid_data[df_sample['is_attack'] == 0].mean()
            attack_mean = valid_data[df_sample['is_attack'] == 1].mean()
            if pd.notna(benign_mean) and pd.notna(attack_mean):
                # Относительная разница
                diff = abs(attack_mean - benign_mean) / (abs(benign_mean) + 1e-10)
                feature_importance[col] = diff

    # Топ-15 признаков
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:15]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=[f[0] for f in top_features],
        x=[f[1] for f in top_features],
        orientation='h',
        marker_color='#3498db'
    ))

    fig.update_layout(
        title='Топ-15 признаков по различию BENIGN vs ATTACK',
        title_x=0.5,
        xaxis_title='Относительная разница средних',
        yaxis_title='Признак',
        height=600,
        margin=dict(l=250)
    )

    return fig


def generate_report(
    audit_results: Dict[str, Any],
    eda_files: Dict[str, Path],
    config: Optional[Dict[str, Any]] = None,
    output_path: Optional[Path] = None
) -> Path:
    """Сгенерировать HTML отчёт"""
    if config is None:
        config = load_config()

    root = get_project_root()
    if output_path is None:
        output_path = root / config["paths"]["reports"]

    ensure_dir(output_path)

    # Определяем расширение файлов
    first_file = list(eda_files.values())[0] if eda_files else None
    img_ext = first_file.suffix if first_file else ".png"

    # Генерируем HTML с встроенными изображениями
    img_tags = ""
    for name, path in eda_files.items():
        if img_ext == ".png":
            img_tags += f'''
            <div class="viz-container">
                <h3>{name.replace("_", " ").title()}</h3>
                <img src="figures/{path.name}" alt="{name}" style="max-width:100%; height:auto;">
            </div>
            '''
        else:
            img_tags += f'''
            <div class="viz-container">
                <h3>{name.replace("_", " ").title()}</h3>
                <iframe src="figures/{path.name}" width="100%" height="600px" frameborder="0"></iframe>
            </div>
            '''

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>CIC-IDS-2017 Data Audit Report</title>
        <meta charset="utf-8">
        <style>
            body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; background: #f0f2f5; }}
            .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #1a73e8; border-bottom: 3px solid #1a73e8; padding-bottom: 15px; }}
            h2 {{ color: #202124; margin-top: 40px; padding-bottom: 10px; border-bottom: 1px solid #dadce0; }}
            h3 {{ color: #5f6368; }}
            .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
            .stat-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 25px; border-radius: 12px; text-align: center; color: white; }}
            .stat-value {{ font-size: 36px; font-weight: bold; }}
            .stat-label {{ opacity: 0.9; margin-top: 5px; font-size: 14px; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #e0e0e0; }}
            th {{ background: #f8f9fa; font-weight: 600; color: #202124; }}
            tr:hover {{ background: #f8f9fa; }}
            .issue {{ background: #fef7e0; padding: 12px 16px; margin: 8px 0; border-left: 4px solid #f9ab00; border-radius: 4px; }}
            .success {{ background: #e6f4ea; padding: 12px 16px; margin: 8px 0; border-left: 4px solid #34a853; border-radius: 4px; }}
            .viz-container {{ margin: 30px 0; padding: 20px; background: #fafafa; border-radius: 8px; }}
            .viz-container img {{ border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
            .summary-box {{ background: #e8f0fe; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 CIC-IDS-2017 Data Audit Report</h1>
            <p><strong>Generated:</strong> {audit_results['timestamp']}</p>
            
            <h2>📈 Основная статистика</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{format_number(audit_results['basic_stats']['total_rows'])}</div>
                    <div class="stat-label">Всего записей</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{audit_results['basic_stats']['total_columns']}</div>
                    <div class="stat-label">Колонок</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{audit_results['basic_stats']['memory_mb']} MB</div>
                    <div class="stat-label">Размер в памяти</div>
                </div>
                <div class="stat-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                    <div class="stat-value">{len(audit_results['issues'])}</div>
                    <div class="stat-label">Проблем найдено</div>
                </div>
            </div>
            
            <h2>🎯 Распределение классов</h2>
            <table>
                <tr><th>Класс</th><th>Количество</th><th>Доля</th></tr>
                {"".join(f"<tr><td>{'🟢 ' if label == 'BENIGN' else '🔴 '}{label}</td><td>{format_number(info['count'])}</td><td>{info['percent']:.2f}%</td></tr>" for label, info in audit_results['target_distribution'].items())}
            </table>
            
            <div class="summary-box">
                <strong>📌 Резюме по классам:</strong>
                <ul>
                    <li>BENIGN (нормальный трафик): {audit_results['target_distribution'].get('BENIGN', {}).get('percent', 0):.1f}%</li>
                    <li>Атаки: {100 - audit_results['target_distribution'].get('BENIGN', {}).get('percent', 0):.1f}%</li>
                    <li>Редкие классы (<100 записей): Heartbleed, Infiltration, SQL Injection</li>
                </ul>
            </div>
            
            <h2>⚠️ Обнаруженные проблемы</h2>
            {"".join(f'<div class="issue">⚠️ {issue}</div>' for issue in audit_results['issues']) if audit_results['issues'] else '<div class="success">✅ Критических проблем не обнаружено!</div>'}
            
            <h2>📊 Визуализации</h2>
            {img_tags}
            
            <h2>📋 Детали качества данных</h2>
            <h3>Пропуски (NaN)</h3>
            <p>Колонок с пропусками: <strong>{audit_results['missing_values'].get('columns_with_nan', 0)}</strong></p>
            
            <h3>Бесконечности (Inf)</h3>
            <p>Колонок с бесконечностями: <strong>{audit_results['infinities'].get('columns_with_inf', 0)}</strong></p>
            
            <h3>Дубликаты</h3>
            <p>Полных дубликатов: <strong>{format_number(audit_results['duplicates']['full_duplicates']['count'])}</strong> ({audit_results['duplicates']['full_duplicates']['percent']}%)</p>
        </div>
    </body>
    </html>
    """

    report_path = output_path / "data_audit_report.html"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    # JSON отчёт
    json_path = output_path / "audit_results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(audit_results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n✅ Report saved to: {report_path}")

    return report_path