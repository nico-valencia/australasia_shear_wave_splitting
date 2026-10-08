'''
May 15 2026 
Final list of stations has been made along with measurements
Dictionary lives here since it is huge 

'sks' = sks phases only 
'sks_deep' = sks and deep S phases (below 400km atm)
'all' = sks, deep S, shallow S 
'complex', data exists but cannot be fit with a single layer SI curve
            data could be complicated or null
'''

STATION_PHASE_MAP = {
    "ARMA": "sks_deep", 
    "AS31": "sks", 
    "BBOO": "sks_deep",
    "BKB": "all",
    "BKNI": "complex",
    "BLDU": "sks",
    "CAN": "sks_deep",
    "CMSA": "complex",
    "CNB": "sks_deep",
    "COEN": "sks",
    "CTA": "complex",
    "CTAO": "complex",
    "EIDS": "sks_deep",
    "FAKI": "complex",
    "FITZ": "sks_deep",
    "FORT": "all",
    "GIRL": "all",
    "GUMO": "complex",
    "HTT": "complex",
    "INKA": "complex",
    "JAGI": "complex",
    "KDU": "complex",
    "KMBL": "complex",
    "KNRA": "complex",
    "LCRK": "complex",
    "LHI": "complex",
    "LUWI": "complex",
    "MANU": "complex",
    "MEEK": "complex",
    "MMRI": "sks_deep",
    "MNAI": "all",
    "MOO": "complex",
    "MORW": "complex",
    "MTN": "sks_deep",
    "MULG": "complex",
    "MUN": "sks_deep",
    "NFK": "complex",
    "NOUC": "sks_deep",
    "NWAO": "complex",
    "OOD": "complex",
    "PMBI": "complex", #potentially a null station?
    "PMG": "complex",
    "QIS": "sks",
    "QLP": "complex",
    "RABL": "complex",
    "RIV": "complex", # another potential null
    "SAUI": "complex",  # all to complex after test4 snr criteria
    "SMRI": "complex",
    "SNZO": "all",
    "SOEI": "complex",
    "STKA": "sks_deep",
    "TNTI": "sks_deep",
    "TOO": "sks_deep",
    "UGM": "complex",
    "WB2": "complex",   # good null
    "WRKA": "complex", # super good null
    "XMI": "all",
    "XMIS": "all",
    "YNG": "sks_deep",
    }