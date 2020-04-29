import os

_data_dir_path = ".\\клинические_данные"  # windows

# sub pathes
_file_mri_ct_data_sbpath = "01_мрт-кт\\00_пациенты_CT_MRI.xlsx"
_file_biopsy_data_sbpath = "02_биопсия\\00_биопсия.данные.xlsx"

_output_mri_ct_dir_sbpath = "01_мрт-кт"
_output_biopsy_data_sbpath = "02_биопсия"

# abs pathes
file_mri_ct_data_path = os.path.join(_data_dir_path, _file_mri_ct_data_sbpath)
file_biopsy_data_path = os.path.join(_data_dir_path, _file_biopsy_data_sbpath)

output_mri_ct_path = os.path.join(_data_dir_path, _output_mri_ct_dir_sbpath)
output_biopsy_data_path = os.path.join(_data_dir_path, _output_biopsy_data_sbpath)
