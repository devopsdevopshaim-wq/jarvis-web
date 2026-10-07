#!/usr/bin/env python3
"""Builds face-mesh.json from a portrait photo (default face.jpg).

MediaPipe finds 478 landmarks on the photo; this script turns them into a triangle mesh that the
page warps in WebGL (jaw, lips, brows, eyelids, head turn) to make the portrait talk, and into
the per-point weights that say how much each point follows each movement.

    pip install mediapipe==0.10.14 opencv-python-headless scipy numpy
    python3 tools/build-face.py face.jpg face-mesh.json
"""
import json
import sys

import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import Delaunay

src = sys.argv[1] if len(sys.argv) > 1 else 'face.jpg'
dst = sys.argv[2] if len(sys.argv) > 2 else 'face-mesh.json'

img = cv2.imread(src)
H, W = img.shape[:2]
fm = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True, max_num_faces=1)
res = fm.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
if not res.multi_face_landmarks:
    sys.exit('no face found in ' + src)
lm = res.multi_face_landmarks[0].landmark[:468]
P = np.array([[p.x * W, p.y * H] for p in lm])
Z = np.array([p.z * W for p in lm])

OUTER_LIP = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
IN_UP = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
IN_LOW = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
INNER_LOOP = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
EYE_R = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
EYE_L = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]
LID_R_UP = [246, 161, 160, 159, 158, 157, 173]; LID_R_LOW = [7, 163, 144, 145, 153, 154, 155]
LID_L_UP = [398, 384, 385, 386, 387, 388, 466]; LID_L_LOW = [382, 381, 380, 374, 373, 390, 249]
BROW_R = [70, 63, 105, 66, 107, 46, 53, 52, 65, 55]
BROW_L = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

mouth_c = P[[61, 291]].mean(axis=0)
mouth_w = np.linalg.norm(P[61] - P[291])
face_h = np.linalg.norm(P[10] - P[152])
face_w = np.linalg.norm(P[234] - P[454])
cx = P[[234, 454]].mean(axis=0)[0]

# anchors: two rings around the face and the image border keep the hair, neck and background still
anch = []
c = P[OVAL].mean(axis=0)
for k in (1.18, 1.42):
    for i in OVAL:
        a = c + (P[i] - c) * k
        anch.append(a)
for t in np.linspace(0, 1, 9):
    anch += [[t * W, 0], [t * W, H], [0, t * H], [W, t * H]]
anch = np.array(anch)
N = len(P)
ALL = np.vstack([P, anch])
ZALL = np.concatenate([Z, np.zeros(len(anch))])

tri = Delaunay(ALL)
tris = [[int(i) for i in t] for t in tri.simplices]

def gauss(pt, centre, sx, sy):
    return float(np.exp(-(((pt[0] - centre[0]) / sx) ** 2 + ((pt[1] - centre[1]) / sy) ** 2)))

lowlip = set(IN_LOW) | {146, 91, 181, 84, 17, 314, 405, 321, 375}
w_jaw = np.zeros(len(ALL)); w_corner = np.zeros(len(ALL)); w_lip = np.zeros(len(ALL))
w_brow = np.zeros(len(ALL)); w_browin = np.zeros(len(ALL)); w_cheek = np.zeros(len(ALL))
chin = P[152]
for i in range(N):
    p = P[i]
    below = p[1] - mouth_c[1]
    if i in lowlip or (below > face_h * 0.02):
        # follows the jaw: strongest at the chin, fades toward the ears and the cheeks
        lat = abs(p[0] - cx) / (face_w * 0.5)
        w = min(1.0, (below + face_h * 0.04) / (chin[1] - mouth_c[1]))
        w *= max(0.0, 1.0 - lat ** 2 * 0.9)
        w_jaw[i] = max(0.0, w)
    if i in lowlip:
        w_jaw[i] = 1.0
    if i in IN_UP or i in {61, 291, 185, 40, 39, 37, 0, 267, 269, 270, 409}:
        pass
    w_corner[i] = max(gauss(p, P[61], mouth_w * 0.45, mouth_w * 0.45), gauss(p, P[291], mouth_w * 0.45, mouth_w * 0.45))
    w_lip[i] = gauss(p, mouth_c, mouth_w * 0.75, face_h * 0.075)
    w_cheek[i] = max(gauss(p, P[50], face_w * 0.14, face_h * 0.08), gauss(p, P[280], face_w * 0.14, face_h * 0.08))
for i in BROW_R + BROW_L:
    w_brow[i] = 1.0
for i in (BROW_R + BROW_L):
    pass
for i in range(N):
    p = P[i]
    forehead = max(gauss(p, P[105], face_w * 0.17, face_h * 0.09), gauss(p, P[334], face_w * 0.17, face_h * 0.09))
    w_brow[i] = max(w_brow[i], forehead * 0.7)
    # inner ends of the brows (toward the nose) for worry and anger
    w_browin[i] = max(gauss(p, P[107], face_w * 0.08, face_h * 0.05), gauss(p, P[336], face_w * 0.08, face_h * 0.05))
for i in IN_UP + IN_LOW + OUTER_LIP:
    w_brow[i] = 0.0

def mask(idx):
    m = np.zeros(len(ALL)); m[idx] = 1; return m

def fix(a):
    return [round(float(x), 3) for x in a]

out = {
    'w': W, 'h': H, 'n': N,
    'pts': [[round(float(x), 1), round(float(y), 1)] for x, y in ALL],
    'z': fix(ZALL / W),
    'tris': [i for t in tris for i in t],
    'w_jaw': fix(w_jaw), 'w_corner': fix(w_corner), 'w_lip': fix(w_lip), 'w_brow': fix(w_brow), 'w_browin': fix(w_browin), 'w_cheek': fix(w_cheek),
    'inner': INNER_LOOP, 'in_up': IN_UP, 'in_low': IN_LOW, 'outer_lip': OUTER_LIP,
    'eye_r': EYE_R, 'eye_l': EYE_L, 'lid_r_up': LID_R_UP, 'lid_r_low': LID_R_LOW, 'lid_l_up': LID_L_UP, 'lid_l_low': LID_L_LOW,
    'mouth_c': [round(float(mouth_c[0]), 1), round(float(mouth_c[1]), 1)], 'mouth_w': round(float(mouth_w), 1),
    'face_h': round(float(face_h), 1), 'face_w': round(float(face_w), 1), 'cx': round(float(cx), 1),
    'pivot': [round(float(P[1][0]), 1), round(float(P[168][1] + face_h * 0.35), 1)],
}
json.dump(out, open(dst, 'w'), separators=(',', ':'))
print('points', len(ALL), 'triangles', len(tris), 'bytes', len(json.dumps(out, separators=(',', ':'))))
