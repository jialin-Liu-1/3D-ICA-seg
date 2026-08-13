import os
import pydicom
import numpy as np
import logging
import re
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set font for better display (supports English)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False


def parse_filename(filename):
    """
    Parse filename, supporting two formats:
    1. ANY_CaseID_ViewDirection_ProjectionNumber
    2. ANY_CaseID_SeriesNumber_ViewDirection_ProjectionNumber

    Returns dictionary with case ID, view direction, and projection number
    """
    # Try to match first format: ANY_CaseID_ViewDirection_ProjectionNumber
    pattern1 = r'ANY_(\d+)_(view\d+)_(\d+)'
    match1 = re.match(pattern1, filename)

    if match1:
        case_id = match1.group(1)
        view_direction = match1.group(2)
        projection_num = match1.group(3)
        return {
            'case_id': case_id,
            'view_direction': view_direction,
            'projection_num': projection_num,
            'has_series_num': False,
            'base_name': f"ANY_{case_id}_{view_direction}_{projection_num}"
        }

    # Try to match second format: ANY_CaseID_SeriesNumber_ViewDirection_ProjectionNumber
    pattern2 = r'ANY_(\d+)_(\d+)_(view\d+)_(\d+)'
    match2 = re.match(pattern2, filename)

    if match2:
        case_id = match2.group(1)
        series_num = match2.group(2)
        view_direction = match2.group(3)
        projection_num = match2.group(4)
        return {
            'case_id': case_id,
            'view_direction': view_direction,
            'projection_num': projection_num,
            'has_series_num': True,
            'series_num': series_num,
            'base_name': f"ANY_{case_id}_{series_num}_{view_direction}_{projection_num}"
        }

    # If neither matches, try generic method with underscore splitting
    parts = filename.split('_')
    if len(parts) >= 4 and parts[0] == 'ANY':
        # Try to determine which format
        if len(parts) == 4:
            # ANY_CaseID_ViewDirection_ProjectionNumber
            case_id = parts[1]
            view_direction = parts[2]
            projection_num = parts[3]
            return {
                'case_id': case_id,
                'view_direction': view_direction,
                'projection_num': projection_num,
                'has_series_num': False,
                'base_name': f"ANY_{case_id}_{view_direction}_{projection_num}"
            }
        elif len(parts) == 5:
            # ANY_CaseID_SeriesNumber_ViewDirection_ProjectionNumber
            case_id = parts[1]
            series_num = parts[2]
            view_direction = parts[3]
            projection_num = parts[4]
            return {
                'case_id': case_id,
                'view_direction': view_direction,
                'projection_num': projection_num,
                'has_series_num': True,
                'series_num': series_num,
                'base_name': f"ANY_{case_id}_{series_num}_{view_direction}_{projection_num}"
            }

    return None


def extract_angle(ds):
    """
    Extract angle information from DICOM header (0018,1510)
    """
    try:
        if hasattr(ds, 'PositionerPrimaryAngle'):
            angle = float(ds.PositionerPrimaryAngle)
            return angle
        elif (0x0018, 0x1510) in ds:
            angle = float(ds[0x0018, 0x1510].value)
            return angle
        else:
            logger.warning("Angle information (0018,1510) not found")
            return None
    except Exception as e:
        logger.error(f"Error extracting angle: {str(e)}")
        return None


def format_angle(angle):
    """
    Format angle: if negative, convert to 360 + angle
    """
    if angle is None:
        return "unknown"

    if angle < 0:
        formatted_angle = int(360 + angle)
    else:
        formatted_angle = int(angle)

    return str(formatted_angle)


def pair_images_by_projection(dicom_files_info):
    """
    Pair view1 and view2 images by case ID and projection number
    Performs many-to-many pairing when possible
    """
    # First step: group by (case_id, projection_num)
    grouped = defaultdict(list)

    for file_info in dicom_files_info:
        pair_key = (file_info['case_id'], file_info['projection_num'])
        grouped[pair_key].append(file_info)

    paired_images = []

    for (case_id, projection_num), files in grouped.items():
        # Separate view1 and view2
        view1_files = [f for f in files if f['view_direction'] == 'view1']
        view2_files = [f for f in files if f['view_direction'] == 'view2']

        if not view1_files or not view2_files:
            logger.warning(f"Case {case_id} projection {projection_num} missing paired images "
                           f"(view1: {len(view1_files)}, view2: {len(view2_files)})")
            continue

        # Check which files have series numbers
        view1_with_series = [f for f in view1_files if f.get('has_series_num', False)]
        view2_with_series = [f for f in view2_files if f.get('has_series_num', False)]

        # Case 1: Both have series numbers, pair by series number
        if view1_with_series and view2_with_series:
            # Group by series number
            view1_by_series = {f['series_num']: f for f in view1_with_series}
            view2_by_series = {f['series_num']: f for f in view2_with_series}

            # Find common series numbers
            common_series = set(view1_by_series.keys()) & set(view2_by_series.keys())

            # Pair by common series numbers
            for series_num in common_series:
                paired_images.append({
                    'case_id': case_id,
                    'projection_num': projection_num,
                    'view1': view1_by_series[series_num],
                    'view2': view2_by_series[series_num],
                    'series_num': series_num
                })

            # Handle unmatched view1
            unmatched_view1 = []
            for f in view1_files:
                if f.get('has_series_num', False) and f['series_num'] not in common_series:
                    unmatched_view1.append(f)
                elif not f.get('has_series_num', False):
                    unmatched_view1.append(f)

            # Handle unmatched view2
            unmatched_view2 = []
            for f in view2_files:
                if f.get('has_series_num', False) and f['series_num'] not in common_series:
                    unmatched_view2.append(f)
                elif not f.get('has_series_num', False):
                    unmatched_view2.append(f)

            # Try to pair remaining view1 and view2 (in order)
            for v1 in unmatched_view1:
                if unmatched_view2:
                    v2 = unmatched_view2.pop(0)
                    series_label = "none"
                    if v1.get('has_series_num', False) and v2.get('has_series_num', False):
                        series_label = f"{v1['series_num']}_{v2['series_num']}"
                    elif v1.get('has_series_num', False):
                        series_label = v1['series_num']
                    elif v2.get('has_series_num', False):
                        series_label = v2['series_num']

                    paired_images.append({
                        'case_id': case_id,
                        'projection_num': projection_num,
                        'view1': v1,
                        'view2': v2,
                        'series_num': series_label
                    })

        else:
            # Case 2: No series numbers or mixed
            # Perform full pairing
            for v1 in view1_files:
                for v2 in view2_files:
                    series_label = "none"
                    if v1.get('has_series_num', False) and v2.get('has_series_num', False):
                        if v1['series_num'] == v2['series_num']:
                            series_label = v1['series_num']
                        else:
                            series_label = f"{v1['series_num']}_{v2['series_num']}"
                    elif v1.get('has_series_num', False):
                        series_label = v1['series_num']
                    elif v2.get('has_series_num', False):
                        series_label = v2['series_num']

                    paired_images.append({
                        'case_id': case_id,
                        'projection_num': projection_num,
                        'view1': v1,
                        'view2': v2,
                        'series_num': series_label
                    })

    logger.info(f"Successfully paired {len(paired_images)} image pairs")
    return paired_images


def analyze_angle_distribution(input_folder, output_folder):
    """
    Analyze angle distribution of paired images
    """
    # Create output folder
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output folder: {output_path}")

    # Get all DICOM files (no extension or .dcm extension)
    dicom_files = []
    for file in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file)
        if os.path.isfile(file_path):
            # Exclude obvious non-DICOM files (like folders)
            if file.endswith('.dcm') or '.' not in file:
                dicom_files.append(file)

    logger.info(f"Found {len(dicom_files)} DICOM files")

    if len(dicom_files) == 0:
        logger.error("No DICOM files found in input folder")
        return

    # Parse all filenames
    files_info = []
    unsupported_files = []
    for filename in dicom_files:
        info = parse_filename(filename)
        if info:
            info['filename'] = filename
            info['full_path'] = os.path.join(input_folder, filename)
            files_info.append(info)
        else:
            unsupported_files.append(filename)

    logger.info(f"Successfully parsed {len(files_info)} filenames")

    if unsupported_files:
        logger.warning(f"Unable to parse {len(unsupported_files)} filenames:")
        for f in unsupported_files[:5]:
            logger.warning(f"  - {f}")
        if len(unsupported_files) > 5:
            logger.warning(f"  ... and {len(unsupported_files) - 5} more files")

    # Pair images
    paired_images = pair_images_by_projection(files_info)
    logger.info(f"Found {len(paired_images)} paired images")

    if len(paired_images) == 0:
        logger.error("No paired images found")
        return

    # Extract angle information
    angle_data = []
    view1_angles = []
    view2_angles = []
    angle_pairs = []

    for pair_info in paired_images:
        case_id = pair_info['case_id']
        projection_num = pair_info['projection_num']
        view1_info = pair_info['view1']
        view2_info = pair_info['view2']
        series_num = pair_info.get('series_num', '')

        try:
            # Read view1
            ds1 = pydicom.dcmread(view1_info['full_path'])
            angle1 = extract_angle(ds1)

            # Read view2
            ds2 = pydicom.dcmread(view2_info['full_path'])
            angle2 = extract_angle(ds2)

            if angle1 is not None and angle2 is not None:
                # Calculate angle difference
                angle_diff = abs(angle1 - angle2)
                # Take the smaller angle (if > 180, use 360 - angle_diff)
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff

                angle_diff = round(angle_diff, 1)

                angle_data.append({
                    'case_id': case_id,
                    'projection_num': projection_num,
                    'series_num': series_num,
                    'view1_angle': angle1,
                    'view2_angle': angle2,
                    'angle_diff': angle_diff
                })

                view1_angles.append(angle1)
                view2_angles.append(angle2)
                angle_pairs.append((angle1, angle2))

                logger.info(
                    f"Case {case_id} Projection {projection_num}: View1={angle1:.1f}°, View2={angle2:.1f}°, Diff={angle_diff:.1f}°")
            else:
                logger.warning(f"Case {case_id} Projection {projection_num} missing angle information")

        except Exception as e:
            logger.error(f"Error processing case {case_id}: {str(e)}")

    logger.info(f"Successfully extracted {len(angle_data)} angle pairs")

    if len(angle_data) == 0:
        logger.error("No angle information extracted")
        return

    # Statistics of angle distribution
    angle_diffs = [item['angle_diff'] for item in angle_data]

    # Create angle bins (10-degree intervals)
    bins = np.arange(0, 185, 10)
    hist, bin_edges = np.histogram(angle_diffs, bins=bins)

    # Calculate percentages
    total = len(angle_diffs)
    percentages = (hist / total * 100).round(2)

    # Create statistics table
    bin_labels = [f"{int(bin_edges[i])}-{int(bin_edges[i + 1])}°" for i in range(len(bin_edges) - 1)]

    df_stats = pd.DataFrame({
        'Angle Range': bin_labels,
        'Count': hist,
        'Percentage (%)': percentages
    })

    # Add cumulative percentage
    df_stats['Cumulative (%)'] = df_stats['Percentage (%)'].cumsum().round(2)

    logger.info("\nAngle Distribution Statistics:")
    logger.info(df_stats.to_string(index=False))

    # Save statistics table to CSV
    csv_path = output_path / 'angle_distribution_stats.csv'
    df_stats.to_csv(csv_path, index=False, encoding='utf-8-sig')
    logger.info(f"Statistics table saved: {csv_path}")

    # Save detailed data to CSV
    df_details = pd.DataFrame(angle_data)
    details_path = output_path / 'angle_details.csv'
    df_details.to_csv(details_path, index=False, encoding='utf-8-sig')
    logger.info(f"Detailed data saved: {details_path}")

    # Create histogram
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot histogram
    bars = ax.bar(bin_labels, hist, color='steelblue', edgecolor='black', alpha=0.7)

    # Display values and percentages on top of bars
    for i, (bar, count, pct) in enumerate(zip(bars, hist, percentages)):
        if count > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                    f'{int(count)}\n({pct:.1f}%)',
                    ha='center', va='bottom', fontsize=9)

    # Set title and labels
    ax.set_xlabel('Angle Range (degrees)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Pairs', fontsize=12, fontweight='bold')
    ax.set_title('DSA Image Pair Angle Distribution', fontsize=14, fontweight='bold')

    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')

    # Add statistics information
    mean_angle = np.mean(angle_diffs)
    median_angle = np.median(angle_diffs)
    std_angle = np.std(angle_diffs)
    max_angle = np.max(angle_diffs)
    min_angle = np.min(angle_diffs)

    stats_text = (f'Total Pairs: {total}\n'
                  f'Mean Angle: {mean_angle:.1f}°\n'
                  f'Median Angle: {median_angle:.1f}°\n'
                  f'Std Dev: {std_angle:.1f}°\n'
                  f'Max Angle: {max_angle:.1f}°\n'
                  f'Min Angle: {min_angle:.1f}°')

    # Add statistics box in upper right corner
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.97, 0.97, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=props)

    plt.tight_layout()

    # Save histogram
    plot_path = output_path / 'angle_distribution_histogram.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Histogram saved: {plot_path}")

    # Close figure to free memory
    plt.close()

    # Output summary
    logger.info("\n" + "=" * 60)
    logger.info("Analysis Complete!")
    logger.info(f"Total paired images: {total}")
    logger.info(f"Angle range: {min_angle:.1f}° - {max_angle:.1f}°")
    logger.info(f"Mean angle: {mean_angle:.1f}° (±{std_angle:.1f}°)")
    logger.info(f"Median angle: {median_angle:.1f}°")
    logger.info(f"Output files:")
    logger.info(f"  - Statistics table: {csv_path}")
    logger.info(f"  - Detailed data: {details_path}")
    logger.info(f"  - Histogram: {plot_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    # Set input and output folder paths
    input_folder = r"F:\view"
    output_folder = r"D:\med_data\biron\data2\sum"

    logger.info("Starting DSA image pair angle distribution analysis...")
    logger.info(f"Input folder: {input_folder}")
    logger.info(f"Output folder: {output_folder}")

    try:
        analyze_angle_distribution(input_folder, output_folder)
        logger.info("Analysis complete!")
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())