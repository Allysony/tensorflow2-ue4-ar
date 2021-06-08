config = {
    'tflite_model': '.\models\posenet_mobilenet_v1.tflite',

    'IMAGEW': 640,
    'IMAGEH': 480,

    'IN_IMAGEW': 257,
    'IN_IMAGEH': 257,

    'num_joints': 17,
    'min_confidence': 0.5,
    'circle_radius': 8,
    'edge_radius': 3,

    'joints': {
        '0': 'Nose',
        '1': 'LEye',
        '2': 'REye',
        '3': 'LEar',
        '4': 'REar',
        '5': 'LShldr',
        '6': 'RShldr',
        '7': 'LElbow',
        '8': 'RElbow',
        '9': 'LWrist',
        '10': 'RWrist',
        '11': 'LHip',
        '12': 'RHip',
        '13': 'LKnee',
        '14': 'RKnee',
        '15': 'LAnkle',
        '16': 'RAnkle'
    },

    'edges': [
        (5, 6),
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (5, 11),
        (6, 12),
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
    ],
}
