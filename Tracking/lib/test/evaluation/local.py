from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/data1/saizhou777/ostrack20251012/data/got10k_lmdb'
    settings.got10k_path = '/data2/xd/dataset/GOT-10k/raw/GOT-10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/data1/saizhou777/ostrack20251012/data/itb'
    settings.lasot_extension_subset_path_path = '/data1/saizhou777/ostrack20251012/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/data1/saizhou777/ostrack20251012/data/lasot_lmdb'
    settings.lasot_path = '/data2/xd/dataset/LaSOT/raw/LaSOT'
    settings.network_path = '/data1/saizhou777/ostrack20251012/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/data1/saizhou777/ostrack20251012/data/nfs'
    settings.otb_path = '/data1/saizhou777/ostrack20251012/data/otb'
    settings.prj_dir = '/data1/saizhou777/ostrack20251012'
    settings.result_plot_path = '/data1/saizhou777/ostrack20251012/output/test/result_plots'
    settings.results_path = '/data1/saizhou777/ostrack20251012/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/data1/saizhou777/ostrack20251012/output'
    settings.segmentation_path = '/data1/saizhou777/ostrack20251012/output/test/segmentation_results'
    settings.tc128_path = '/data1/saizhou777/ostrack20251012/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/data1/saizhou777/ostrack20251012/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/data2/xd/dataset/TrackingNet/raw/TrackingNet'
    settings.uav_path = '/data2/saizhou777/database/uav'
    settings.uavtrack112_path = '/data1/saizhou777/ostrack20251012/data/uavtrack112'
    settings.utb180_path = '/data1/saizhou777/ostrack20251012/data/utb180'
    settings.vmat_path = '/data1/saizhou777/ostrack20251012/data/vmat'
    settings.vot18_path = '/data1/saizhou777/ostrack20251012/data/vot2018'
    settings.vot22_path = '/data1/saizhou777/ostrack20251012/data/vot2022'
    settings.vot_path = '/data1/saizhou777/ostrack20251012/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

