def check_files_if_exist(*paths):
    from pathlib import Path

    try:
        all(Path(path).resolve(strict=True) for path in paths)
    except FileNotFoundError:
        print("File OR Folder:", "Not exist")
        raise
    else:
        # Exist
        return True


def project_quadpol_rat_files(
    DIR_campaign, ID_campaign, ID_flight, band, ID_Pass, n_try="01"
):
    """Dictionary of Full Polarimetric SAR RAT files

    Provide a dic of str for RAT files of each channel (HH, VV, HV, VH)

    Parameters
    ----------
    DIR_campaign : str
        direcotry of the campaign data
    ID_campaign : str
        ID for the campaign
    ID_flight : str
        ID for the flight
    band : str
        band of F-SAR bands=['X','C','S','L']
    ID_Pass : str
        ID pass of a given flight path -> general master (processing master)
    n_try : str, optional
        processing trial ("number after t*" in filename)

    Returns
    -------
    files : dict
        dict of RAT files "RAT"

    Examples
    --------
    >>> DIR_campaign = "/data/largeHome/mans_is/PolSAR-Thesis/10_users/mans_is/PolSAR-Results/01_projects/18PRMASR"
    >>> ID_campaign = '18prmasr'
    >>> ID_flight="03"
    >>> band="L"
    >>> ID_Pass="05"
    >>> n_try="01"
    >>> project_quadpol_rat_files(DIR_campaign=DIR_campaign, ID_campaign=ID_campaign, ID_flight=ID_flight, band=band, ID_Pass=ID_Pass, n_try="01")
    {'AOI': '/data/HR_Projekte/Pol-InSAR_InfoRetrieval/01_projects/18PRMASR/FL03/PS05/T01L/RGI/RGI-SR/incidence_18prmasr0305_L_t01L.rat',
        'HH': '/data/HR_Projekte/Pol-InSAR_InfoRetrieval/01_projects/18PRMASR/FL03/PS05/T01L/RGI/RGI-SR/slc_18prmasr0305_Lhh_t01L.rat',
        'VV': '/data/HR_Projekte/Pol-InSAR_InfoRetrieval/01_projects/18PRMASR/FL03/PS05/T01L/RGI/RGI-SR/slc_18prmasr0305_Lvv_t01L.rat',
        'HV': '/data/HR_Projekte/Pol-InSAR_InfoRetrieval/01_projects/18PRMASR/FL03/PS05/T01L/RGI/RGI-SR/slc_18prmasr0305_Lhv_t01L.rat',
        'VH': '/data/HR_Projekte/Pol-InSAR_InfoRetrieval/01_projects/18PRMASR/FL03/PS05/T01L/RGI/RGI-SR/slc_18prmasr0305_Lvh_t01L.rat'}
    """
    from pathlib import Path

    try:
        DIR_campaign = Path(DIR_campaign).resolve(
            strict=True
        )  # .exists() # .resolve(strict=True)
        DIR_data = Path.joinpath(
            DIR_campaign,
            "FL"
            + ID_flight
            + "/"
            + "PS"
            + ID_Pass
            + "/T01"
            + band
            + "/RGI/RGI-SR/",
        )
        DIR_data = Path(DIR_data).resolve(strict=True)
    except FileNotFoundError:
        # DIR_data doesn't exist
        print("DIR_data: ", DIR_data, "Not exist")
        raise
    else:
        # exists
        hh_file = Path.joinpath(
            DIR_data,
            "slc_"
            + ID_campaign
            + ID_flight
            + ID_Pass
            + "_"
            + band
            + "hh_t"
            + n_try
            + band
            + ".rat",
        )
        vv_file = Path.joinpath(
            DIR_data,
            "slc_"
            + ID_campaign
            + ID_flight
            + ID_Pass
            + "_"
            + band
            + "vv_t"
            + n_try
            + band
            + ".rat",
        )
        hv_file = Path.joinpath(
            DIR_data,
            "slc_"
            + ID_campaign
            + ID_flight
            + ID_Pass
            + "_"
            + band
            + "hv_t"
            + n_try
            + band
            + ".rat",
        )
        vh_file = Path.joinpath(
            DIR_data,
            "slc_"
            + ID_campaign
            + ID_flight
            + ID_Pass
            + "_"
            + band
            + "vh_t"
            + n_try
            + band
            + ".rat",
        )
        aoi_file = Path.joinpath(
            DIR_data,
            "incidence_"
            + ID_campaign
            + ID_flight
            + ID_Pass
            + "_"
            + band
            + "_t"
            + n_try
            + band
            + ".rat",
        )
        check_files_if_exist(hh_file, vv_file, vh_file, vh_file, aoi_file)
        files = {
            "AOI": str(aoi_file),
            "HH": str(hh_file),
            "VV": str(vv_file),
            "HV": str(hv_file),
            "VH": str(vh_file),
        }
        return files


def stand_pol_rat_files(
    DIR_campaign, ID_campaign, ID_flight, band, ID_Pass, n_try="01"
):
    """Dictionary of Full Polarimetric SAR RAT files with regex

    Provide a dic of str for RAT files of each channel (AOI, HH, VV, HV, VH) with Regex, this aviod typical error in file name

    Parameters
    ----------
    DIR_campaign : str
        direcotry of the campaign data
    ID_campaign : str
        ID for the campaign
    ID_flight : str
        ID for the flight
    band : str
        band of F-SAR bands=['X','C','S','L']
    ID_Pass : str
        ID pass of a given flight path -> general master (processing master)
    n_try : str, optional
        processing trial ("number after t*" in filename)

    Returns
    -------
    files : dict
        dict of RAT files "RAT"

    Examples
    --------
    >>> DIR_campaign = "/data/largeHome/mans_is/PolSAR-Thesis/10_users/mans_is/PolSAR-Results/01_projects/18PRMASR"
    >>> ID_campaign = '18prmasr'
    >>> ID_flight="03"
    >>> band="L"
    >>> ID_Pass="05"
    >>> n_try="01"
    >>> project_quadpol_rat_files(DIR_campaign=DIR_campaign, ID_campaign=ID_campaign, ID_flight=ID_flight, band=band, ID_Pass=ID_Pass, n_try="01")
    {'AOI': '/data/HR_Projekte/Pol-InSAR_InfoRetrieval/01_projects/18PRMASR/FL03/PS05/T01L/RGI/RGI-SR/incidence_18prmasr0305_L_t01L.rat',
        'HH': '/data/HR_Projekte/Pol-InSAR_InfoRetrieval/01_projects/18PRMASR/FL03/PS05/T01L/RGI/RGI-SR/slc_18prmasr0305_Lhh_t01L.rat',
        'VV': '/data/HR_Projekte/Pol-InSAR_InfoRetrieval/01_projects/18PRMASR/FL03/PS05/T01L/RGI/RGI-SR/slc_18prmasr0305_Lvv_t01L.rat',
        'HV': '/data/HR_Projekte/Pol-InSAR_InfoRetrieval/01_projects/18PRMASR/FL03/PS05/T01L/RGI/RGI-SR/slc_18prmasr0305_Lhv_t01L.rat',
        'VH': '/data/HR_Projekte/Pol-InSAR_InfoRetrieval/01_projects/18PRMASR/FL03/PS05/T01L/RGI/RGI-SR/slc_18prmasr0305_Lvh_t01L.rat'}
    """
    from pathlib import Path

    try:
        DIR_campaign = Path(DIR_campaign).resolve(
            strict=True
        )  # .exists() # .resolve(strict=True)
        DIR_data = Path.joinpath(
            DIR_campaign,
            "FL"
            + ID_flight
            + "/"
            + "PS"
            + ID_Pass
            + "/T01"
            + band
            + "/RGI/RGI-SR/",
        )
        DIR_data = Path(DIR_data).resolve(strict=True)
    except FileNotFoundError:
        # DIR_data doesn't exist
        print("DIR_data: ", DIR_data, "Not exist")
        raise
    else:
        # exists
        hh_file = list(DIR_data.glob("slc_*" + "hh" + "*.rat"))[0]

        vv_file = list(DIR_data.glob("slc_*" + "vv" + "*.rat"))[0]
        hv_file = list(DIR_data.glob("slc_*" + "hv" + "*.rat"))[0]
        vh_file = list(DIR_data.glob("slc_*" + "vh" + "*.rat"))[0]

        aoi_file = list(DIR_data.glob("incidence_" + "*.rat"))[0]

        check_files_if_exist(hh_file, vv_file, vh_file, vh_file, aoi_file)
        files = {
            "AOI": str(aoi_file),
            "HH": str(hh_file),
            "VV": str(vv_file),
            "HV": str(hv_file),
            "VH": str(vh_file),
        }
        return files
