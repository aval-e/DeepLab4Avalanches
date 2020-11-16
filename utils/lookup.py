# some constants and lookup tables
TYP_2_INSTLABEL = {'BACKGROUND': 0,
                   'UNKNOWN': 1,
                   'SLAB': 2,
                   'LOOSE_SNOW': 3,
                   'FULL_DEPTH': 4}

INSTLABEL_2_STR = {v: k for k, v in TYP_2_INSTLABEL.items()}

# avalanche ground truth mapping status
STATUS_2_STR = ['null', 'True', 'Unkown', 'False', '4', 'Old']
# colors for plotting
STATUS_COLORS = ['b', 'g', 'b', 'r', 'b', 'y']
