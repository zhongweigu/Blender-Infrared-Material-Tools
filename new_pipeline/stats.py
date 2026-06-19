import numpy as np
from new_pipeline import config


def print_part_stats(obj_name, temperatures):
    """Print min / max / mean temperature for one part."""
    if temperatures is None or len(temperatures) == 0:
        print(f"  {obj_name:12s} | (无数据)")
        return

    arr = np.asarray(temperatures)
    print(f"  {obj_name:12s} | min={arr.min():8.2f} K  max={arr.max():8.2f} K  mean={arr.mean():8.2f} K")


def compute_engine_temperature(engine_obj):
    """Compute per-face temperature for an engine mesh.

    Currently returns uniform temperature based on config.T_ENGINE_INIT.
    Future: may add gradient from exhaust to front.
    """
    if engine_obj is None or engine_obj.type != 'MESH':
        return None
    mesh = engine_obj.data
    n = len(mesh.polygons)
    return np.full(n, config.T_ENGINE_INIT)


def print_aircraft_summary(T_aircraft, source_faces, iterations, max_change, aircraft_name="Aircraft"):
    """Print summary block for the aircraft diffusion result."""
    arr = np.asarray(T_aircraft)
    valid = arr[~np.isnan(arr)]

    print()
    print("=" * 60)
    print("  稳态温度计算结果")
    print("=" * 60)
    print(f"  蒙皮物体: {aircraft_name}")
    print(f"  热源面片数: {len(source_faces)}")
    print(f"  扩散迭代次数: {iterations}")
    print(f"  最终最大温度变化: {max_change:.6f} K")
    print()

    print(f"  {'部位':12s} | {'最低温':>10s}  {'最高温':>10s}  {'平均温':>10s}")
    print(f"  {'-' * 12}-+-{'-' * 10}--{'-' * 10}--{'-' * 10}")

    # Aircraft overall
    print_part_stats(aircraft_name, valid)

    return valid.min(), valid.max(), valid.mean()


def print_heat_source_summary(T_source_dict):
    """Print T_s range for heat source faces."""
    if not T_source_dict:
        print("  未找到热源面片")
        return

    vals = np.array(list(T_source_dict.values()))
    print()
    print(f"  热源面片 T_s 范围: min={vals.min():.2f} K  max={vals.max():.2f} K  mean={vals.mean():.2f} K")


def print_all_stats(T_aircraft, source_faces, iterations, max_change,
                    eng_data_l, eng_data_r, T_source_dict,
                    aircraft_name="Aircraft", eng_l_name="Engin_L", eng_r_name="Engin_R"):
    """Print all temperature statistics."""
    # Aircraft stats
    print_aircraft_summary(T_aircraft, source_faces, iterations, max_change, aircraft_name)

    # Engine stats
    print_part_stats(eng_l_name, eng_data_l)
    print_part_stats(eng_r_name, eng_data_r)

    # Heat source detail
    print_heat_source_summary(T_source_dict)

    # Source face T_s on aircraft
    if source_faces and T_aircraft is not None:
        source_temps = np.array([T_aircraft[i] for i in source_faces])
        print(f"  热源面片上机身后温度: min={source_temps.min():.2f} K  max={source_temps.max():.2f} K  mean={source_temps.mean():.2f} K")

    print()
    print("=" * 60)
    print()

    # Per-part summary table
    print("  各部位温度汇总:")
    print(f"  {'部位':12s} | {'最低温':>10s}  {'最高温':>10s}  {'平均温':>10s}")
    print(f"  {'-' * 12}-+-{'-' * 10}--{'-' * 10}--{'-' * 10}")
    print_part_stats(aircraft_name, T_aircraft)
    print_part_stats(eng_l_name, eng_data_l)
    print_part_stats(eng_r_name, eng_data_r)
    print()
