"""
Script para analizar estadísticas de conteo de personas

Este script lee los archivos JSON/CSV generados por el sistema de conteo
y genera visualizaciones y reportes detallados.
"""

import os
import json
import csv
from datetime import datetime
from collections import defaultdict
import glob

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("⚠️ Matplotlib no disponible. Instalar con: pip install matplotlib")
    MATPLOTLIB_AVAILABLE = False

from config import DATA_DIR

STATS_DIR = os.path.join(DATA_DIR, "estadisticas")


def list_available_stats():
    """Lista todos los archivos de estadísticas disponibles"""
    if not os.path.exists(STATS_DIR):
        print(f"❌ No existe el directorio: {STATS_DIR}")
        return []

    json_files = glob.glob(os.path.join(STATS_DIR, "stats_*.json"))

    if not json_files:
        print("❌ No se encontraron archivos de estadísticas")
        return []

    print(f"\n📊 Archivos de estadísticas encontrados: {len(json_files)}")
    print("="*70)

    files_info = []
    for i, filepath in enumerate(sorted(json_files, reverse=True)):
        filename = os.path.basename(filepath)
        # Extraer timestamp del nombre
        timestamp_str = filename.replace("stats_", "").replace(".json", "")
        try:
            dt = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            date_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            date_str = "Fecha desconocida"

        file_size = os.path.getsize(filepath) / 1024  # KB

        files_info.append({
            "index": i,
            "path": filepath,
            "filename": filename,
            "date": date_str,
            "size_kb": file_size
        })

        print(f"{i}. {filename}")
        print(f"   Fecha: {date_str}")
        print(f"   Tamaño: {file_size:.1f} KB")
        print()

    return files_info


def load_stats_file(filepath):
    """Carga un archivo de estadísticas JSON"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"❌ Error cargando {filepath}: {e}")
        return None


def print_summary(stats_data):
    """Imprime resumen de estadísticas"""
    summary = stats_data.get("summary", {})

    print("\n" + "="*70)
    print("📊 RESUMEN DE SESIÓN")
    print("="*70)

    # Información de sesión
    session = summary.get("session", {})
    print(f"\n⏱️  DURACIÓN DE SESIÓN:")
    print(f"   Inicio: {session.get('start_time', 'N/A')}")
    print(f"   Duración: {session.get('duration_seconds', 0):.1f} segundos")
    print(f"   Frames procesados: {session.get('frames_processed', 0)}")

    # Estado actual
    current = summary.get("current", {})
    print(f"\n👥 ESTADO FINAL:")
    print(f"   Total en área: {current.get('total_people', 0)}")
    print(f"   Identificados: {current.get('identified', 0)}")
    print(f"   Desconocidos: {current.get('unknown', 0)}")

    # Personas únicas
    unique = summary.get("unique", {})
    print(f"\n🔢 PERSONAS ÚNICAS:")
    print(f"   Total vistas: {unique.get('total_unique_tracks', 0)}")
    print(f"   Identificadas: {unique.get('unique_identified_people', 0)}")
    print(f"   Desconocidas: {unique.get('unique_unknown', 0)}")

    # Nombres identificados
    names = unique.get('identified_names', [])
    if names:
        print(f"\n📝 PERSONAS IDENTIFICADAS:")
        for name in names:
            print(f"   - {name}")

    # Pico
    peak = summary.get("peak", {})
    print(f"\n📈 PICO DE PERSONAS:")
    print(f"   Máximo simultáneas: {peak.get('max_people_at_once', 0)}")
    print(f"   Momento: {peak.get('peak_time', 'N/A')}")

    # Zonas (si existen)
    zones = current.get("zones", {})
    if zones:
        print(f"\n🗺️  CONTEO POR ZONAS:")
        for zone_name, count in zones.items():
            print(f"   {zone_name}: {count} personas")

    print("="*70)


def analyze_history(stats_data):
    """Analiza el historial temporal"""
    history = stats_data.get("history", [])

    if not history:
        print("\n⚠️ No hay datos de historial temporal")
        return

    print(f"\n📈 ANÁLISIS DE HISTORIAL ({len(history)} registros)")
    print("="*70)

    # Calcular promedios
    total_sum = sum(h.get('total', 0) for h in history)
    identified_sum = sum(h.get('identified', 0) for h in history)
    unknown_sum = sum(h.get('unknown', 0) for h in history)

    avg_total = total_sum / len(history) if history else 0
    avg_identified = identified_sum / len(history) if history else 0
    avg_unknown = unknown_sum / len(history) if history else 0

    print(f"Promedio de personas en área: {avg_total:.1f}")
    print(f"Promedio identificadas: {avg_identified:.1f}")
    print(f"Promedio desconocidas: {avg_unknown:.1f}")

    # Encontrar picos
    max_entry = max(history, key=lambda x: x.get('total', 0))
    print(f"\nPico de personas: {max_entry.get('total', 0)} en {max_entry.get('timestamp', 'N/A')}")


def analyze_people_log(stats_data):
    """Analiza el log detallado de personas"""
    people_log = stats_data.get("people_log", [])

    if not people_log:
        print("\n⚠️ No hay log detallado de personas")
        return

    print(f"\n📋 LOG DE DETECCIONES ({len(people_log)} registros)")
    print("="*70)

    # Contar por identidad
    identity_counts = defaultdict(int)
    identity_confidences = defaultdict(list)

    for entry in people_log:
        identity = entry.get('identity', 'Desconocido')
        confidence = entry.get('confidence', 0)

        identity_counts[identity] += 1
        identity_confidences[identity].append(confidence)

    print("\nDetecciones por persona:")
    for identity, count in sorted(identity_counts.items(), key=lambda x: x[1], reverse=True):
        avg_conf = sum(identity_confidences[identity]) / len(identity_confidences[identity])
        print(f"   {identity}: {count} detecciones (confianza promedio: {avg_conf:.2f})")


def plot_history(stats_data, output_path=None):
    """Genera gráficas del historial"""
    if not MATPLOTLIB_AVAILABLE:
        print("\n⚠️ Matplotlib no disponible. No se pueden generar gráficas.")
        return

    history = stats_data.get("history", [])
    if not history:
        print("\n⚠️ No hay datos para graficar")
        return

    # Extraer datos
    timestamps = []
    totals = []
    identifieds = []
    unknowns = []

    for entry in history:
        try:
            ts = datetime.fromisoformat(entry['timestamp'])
            timestamps.append(ts)
            totals.append(entry.get('total', 0))
            identifieds.append(entry.get('identified', 0))
            unknowns.append(entry.get('unknown', 0))
        except:
            continue

    if not timestamps:
        print("\n⚠️ No se pudieron procesar las fechas")
        return

    # Crear figura
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('Análisis de Conteo de Personas', fontsize=16, fontweight='bold')

    # Gráfica 1: Total de personas en el tiempo
    ax1.plot(timestamps, totals, 'b-', linewidth=2, label='Total')
    ax1.fill_between(timestamps, totals, alpha=0.3)
    ax1.set_ylabel('Personas en Área', fontsize=12)
    ax1.set_title('Total de Personas en el Área', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Formato de fechas
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # Gráfica 2: Identificados vs Desconocidos
    ax2.plot(timestamps, identifieds, 'g-', linewidth=2, label='Identificados')
    ax2.plot(timestamps, unknowns, 'r-', linewidth=2, label='Desconocidos')
    ax2.fill_between(timestamps, identifieds, alpha=0.3, color='green')
    ax2.fill_between(timestamps, unknowns, alpha=0.3, color='red')
    ax2.set_xlabel('Tiempo', fontsize=12)
    ax2.set_ylabel('Número de Personas', fontsize=12)
    ax2.set_title('Personas Identificadas vs Desconocidas', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Formato de fechas
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()

    # Guardar o mostrar
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 Gráfica guardada en: {output_path}")
    else:
        plt.show()


def generate_report(stats_data, output_path):
    """Genera un reporte en texto"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("REPORTE DE CONTEO DE PERSONAS - DRON DJI FPV\n")
        f.write("="*70 + "\n\n")

        summary = stats_data.get("summary", {})

        # Sesión
        session = summary.get("session", {})
        f.write("INFORMACIÓN DE SESIÓN\n")
        f.write("-"*70 + "\n")
        f.write(f"Inicio: {session.get('start_time', 'N/A')}\n")
        f.write(f"Duración: {session.get('duration_seconds', 0):.1f} segundos\n")
        f.write(f"Frames procesados: {session.get('frames_processed', 0)}\n\n")

        # Estadísticas
        unique = summary.get("unique", {})
        f.write("ESTADÍSTICAS DE PERSONAS ÚNICAS\n")
        f.write("-"*70 + "\n")
        f.write(f"Total de personas vistas: {unique.get('total_unique_tracks', 0)}\n")
        f.write(f"Personas identificadas: {unique.get('unique_identified_people', 0)}\n")
        f.write(f"Personas desconocidas: {unique.get('unique_unknown', 0)}\n\n")

        # Nombres
        names = unique.get('identified_names', [])
        if names:
            f.write("PERSONAS IDENTIFICADAS\n")
            f.write("-"*70 + "\n")
            for name in names:
                f.write(f"  - {name}\n")
            f.write("\n")

        # Pico
        peak = summary.get("peak", {})
        f.write("PICO DE ACTIVIDAD\n")
        f.write("-"*70 + "\n")
        f.write(f"Máximo de personas simultáneas: {peak.get('max_people_at_once', 0)}\n")
        f.write(f"Momento del pico: {peak.get('peak_time', 'N/A')}\n\n")

        f.write("="*70 + "\n")

    print(f"\n📄 Reporte guardado en: {output_path}")


def main():
    print("="*70)
    print("📊 ANALIZADOR DE ESTADÍSTICAS - SISTEMA DE CONTEO")
    print("="*70)

    # Listar archivos disponibles
    files = list_available_stats()

    if not files:
        return

    # Seleccionar archivo
    print("\nSelecciona el archivo a analizar (número) o 'q' para salir:")
    try:
        choice = input("> ").strip()

        if choice.lower() == 'q':
            return

        file_index = int(choice)

        if file_index < 0 or file_index >= len(files):
            print("❌ Índice inválido")
            return

        selected_file = files[file_index]
        print(f"\n✅ Analizando: {selected_file['filename']}")

    except ValueError:
        print("❌ Entrada inválida")
        return

    # Cargar datos
    stats_data = load_stats_file(selected_file['path'])

    if not stats_data:
        return

    # Mostrar resumen
    print_summary(stats_data)

    # Análisis de historial
    analyze_history(stats_data)

    # Log de personas
    analyze_people_log(stats_data)

    # Opciones adicionales
    print("\n" + "="*70)
    print("OPCIONES ADICIONALES:")
    print("  1. Generar gráficas (requiere matplotlib)")
    print("  2. Generar reporte en texto")
    print("  3. Ambos")
    print("  q. Salir")
    print("="*70)

    try:
        option = input("\nSelecciona una opción: ").strip()

        if option == '1' or option == '3':
            output_plot = selected_file['path'].replace('.json', '_grafica.png')
            plot_history(stats_data, output_plot)

        if option == '2' or option == '3':
            output_report = selected_file['path'].replace('.json', '_reporte.txt')
            generate_report(stats_data, output_report)

    except KeyboardInterrupt:
        print("\n\n👋 Saliendo...")


if __name__ == "__main__":
    main()
