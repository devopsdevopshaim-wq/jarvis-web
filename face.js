/* A talking portrait.
   The photo is textured on a triangle mesh (face-mesh.json, built from the photo's 478 face landmarks by
   tools/build-face.py). Each frame the mesh points move a little: the jaw and lips follow the speech, the
   corners of the mouth smile, the brows rise or knit, the eyelids blink, and the head sways and turns.
   The inside of the mouth is a separate layer under the face, seen through the hole the open lips make.
   Everything runs in the browser, with no service and no cost.

     const f = new TalkingFace(stage, { image: 'face.jpg', mesh: 'face-mesh.json' });
     await f.ready;
     f.set({ jaw: .6, smile: .3 });            // direct control, any of: jaw wide round smile brow browIn blink yaw pitch roll
     f.mood('happy' | 'calm' | 'curious' | 'concerned' | 'think' | 'listen');
     f.speak(level) // 0..1 mouth energy, call every frame while a voice is playing; f.shape(wide, round) for the vowel
*/
(function (root) {
  'use strict';

  var VS = 'attribute vec2 p;attribute vec2 uv;uniform vec2 res;varying vec2 v;' +
    'void main(){v=uv;vec2 c=(p/res)*2.0-1.0;gl_Position=vec4(c.x,-c.y,0.0,1.0);}';
  var FS = 'precision mediump float;uniform sampler2D tex;varying vec2 v;uniform float warm;' +
    'void main(){vec4 c=texture2D(tex,v);c.rgb*=warm;gl_FragColor=vec4(c.rgb,1.0);}';

  function shader(gl, type, src) {
    var s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    return s;
  }
  function load(src) {
    return new Promise(function (ok, no) { var i = new Image(); i.onload = function () { ok(i); }; i.onerror = no; i.src = src; });
  }
  function clamp(x, a, b) { return Math.max(a, Math.min(b, x)); }

  var MOODS = {
    calm:      { smile: 0.12, brow: 0.0,  browIn: 0.0,  tilt: 0 },
    happy:     { smile: 0.75, brow: 0.2,  browIn: 0.0,  tilt: 0.02 },
    curious:   { smile: 0.1,  brow: 0.75, browIn: 0.0,  tilt: 0.05 },
    concerned: { smile: -0.15, brow: 0.2, browIn: 0.8,  tilt: -0.03 },
    think:     { smile: 0.0,  brow: 0.45, browIn: 0.15, tilt: 0.06 },
    listen:    { smile: 0.3,  brow: 0.25, browIn: 0.0,  tilt: 0.04 }
  };

  function TalkingFace(host, opts) {
    var me = this;
    opts = opts || {};
    this.host = host;
    this.t = 0; this.last = 0;
    this.p = {}; this.target = {}; this.vel = {};
    ['jaw', 'wide', 'round', 'smile', 'brow', 'browIn', 'blink', 'yaw', 'pitch', 'roll', 'breath'].forEach(function (k) { me.p[k] = 0; me.target[k] = 0; me.vel[k] = 0; });
    this.energy = 0; this.speaking = false; this.nextBlink = 1.2; this.blinkAt = -1; this.mood_ = 'calm'; this.nod = 0; this.sway = Math.random() * 10;
    this.ready = Promise.all([load(opts.image || 'face.jpg'), fetch(opts.mesh || 'face-mesh.json').then(function (r) { return r.json(); })]).then(function (r) { me.init(r[0], r[1]); return me; });
  }

  TalkingFace.prototype.init = function (img, m) {
    var host = this.host, me = this;
    this.m = m; this.img = img;
    var W = m.w, H = m.h;
    // layer 1: the face (WebGL). layer 2: the inside of the mouth and the lashes (2D), drawn over it
    var under = document.createElement('canvas'), top = document.createElement('canvas'), over = document.createElement('canvas');
    [under, top, over].forEach(function (c) { c.width = W * 2; c.height = H * 2; c.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;display:block'; host.appendChild(c); });
    this.under = under; this.top = top; this.over = over; this.octx = over.getContext('2d');
    var gl = top.getContext('webgl', { premultipliedAlpha: true, alpha: true, antialias: true }) || top.getContext('experimental-webgl');
    if (!gl) { this.fallback(img); return; }
    this.gl = gl;
    var prog = gl.createProgram();
    gl.attachShader(prog, shader(gl, gl.VERTEX_SHADER, VS)); gl.attachShader(prog, shader(gl, gl.FRAGMENT_SHADER, FS));
    gl.linkProgram(prog); gl.useProgram(prog); this.prog = prog;
    var n = m.pts.length;
    this.pos = new Float32Array(n * 2);
    this.base = new Float32Array(n * 2);
    var uv = new Float32Array(n * 2);
    for (var i = 0; i < n; i++) { this.base[i * 2] = m.pts[i][0]; this.base[i * 2 + 1] = m.pts[i][1]; uv[i * 2] = m.pts[i][0] / W; uv[i * 2 + 1] = m.pts[i][1] / H; }
    this.pos.set(this.base);
    this.posBuf = gl.createBuffer();
    var uvBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, uvBuf); gl.bufferData(gl.ARRAY_BUFFER, uv, gl.STATIC_DRAW);
    var aUV = gl.getAttribLocation(prog, 'uv'); gl.enableVertexAttribArray(aUV); gl.vertexAttribPointer(aUV, 2, gl.FLOAT, false, 0, 0);
    this.aP = gl.getAttribLocation(prog, 'p');
    var idx = gl.createBuffer(); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, idx);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(m.tris), gl.STATIC_DRAW);
    this.nTri = m.tris.length;
    var tex = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.uniform2f(gl.getUniformLocation(prog, 'res'), W, H);
    this.uWarm = gl.getUniformLocation(prog, 'warm');
    gl.viewport(0, 0, top.width, top.height);
    gl.clearColor(0, 0, 0, 0);
    // the colour of the lips and the skin, sampled once, for the blink and the mouth
    this.sample(img);
    this.frame = this.frame.bind(this);
    requestAnimationFrame(this.frame);
  };

  // no WebGL: show the still photo (nothing pretends to move)
  TalkingFace.prototype.fallback = function (img) {
    this.top.remove();
    this.under.getContext('2d').drawImage(img, 0, 0, this.under.width, this.under.height);
    this.still = true;
  };

  TalkingFace.prototype.sample = function (img) {
    var c = document.createElement('canvas'); c.width = this.m.w; c.height = this.m.h;
    var x = c.getContext('2d'); x.drawImage(img, 0, 0);
    var m = this.m;
    function px(i) { var d = x.getImageData(Math.round(m.pts[i][0]), Math.round(m.pts[i][1]), 1, 1).data; return [d[0], d[1], d[2]]; }
    var lip = px(14), lid = px(159), skin = px(205);
    this.lipCol = lip; this.skin = skin; this.lid = lid;
  };

  // ---- control
  TalkingFace.prototype.set = function (o) { for (var k in o) if (k in this.target) this.target[k] = o[k]; };
  TalkingFace.prototype.mood = function (name) {
    var md = MOODS[name] || MOODS.calm; this.mood_ = name;
    this.base_ = md;
  };
  // mouth energy 0..1 and vowel shape: wide (ee) vs round (oo)
  TalkingFace.prototype.speak = function (level, wide, round) {
    this.energy = clamp(level, 0, 1);
    this.target.jaw = this.energy * 0.95;
    if (wide != null) this.target.wide = wide;
    if (round != null) this.target.round = round;
  };
  TalkingFace.prototype.stopSpeaking = function () { this.energy = 0; this.target.jaw = 0; this.target.wide = 0; this.target.round = 0; };
  TalkingFace.prototype.nodOnce = function (a) { this.nod = a == null ? 1 : a; };
  TalkingFace.prototype.blinkNow = function () { this.blinkAt = this.t; };

  // critically damped follow, so nothing jumps
  function follow(me, k, dt, w) {
    var x = me.p[k], v = me.vel[k], t = me.target[k];
    var d = x - t, e = Math.exp(-w * dt), q = (v + w * d) * dt;
    me.p[k] = t + (d + q) * e;
    me.vel[k] = (v - w * q) * e;
  }

  TalkingFace.prototype.frame = function (now) {
    var me = this, dt = Math.min(0.05, (now - (this.last || now)) / 1000); this.last = now; this.t += dt;
    if (!document.hidden) this.update(dt);
    requestAnimationFrame(this.frame);
  };

  TalkingFace.prototype.update = function (dt) {
    var t = this.t, tg = this.target, md = this.base_ || MOODS.calm;
    // mood is the base; speech adds to it
    tg.smile = md.smile + (this.speaking ? this.target.smileExtra || 0 : 0);
    tg.brow = Math.max(md.brow, this.browKick || 0);
    tg.browIn = md.browIn;
    this.browKick = Math.max(0, (this.browKick || 0) - dt * 1.5);
    // idle life: breathing and a slow sway of the head
    var s = this.sway;
    tg.breath = Math.sin(t * 1.5) * 0.5 + 0.5;
    var yawBase = Math.sin(t * 0.37 + s) * 0.35 + Math.sin(t * 0.91 + s * 2) * 0.15;
    var pitchBase = Math.sin(t * 0.43 + s * 3) * 0.25 + md.tilt * 2;
    var rollBase = Math.sin(t * 0.29 + s * 5) * 0.25 + md.tilt * 3;
    this.nod = Math.max(0, this.nod - dt * 2.2);
    var nodv = Math.sin((1 - this.nod) * Math.PI) * this.nod;
    tg.yaw = yawBase * (this.speaking ? 1.3 : 1);
    tg.pitch = pitchBase + nodv * 1.4 + (this.speaking ? this.energy * 0.25 : 0);
    tg.roll = rollBase;
    // blink at human intervals, with an occasional double blink
    if (t > this.nextBlink && this.blinkAt < 0) { this.blinkAt = t; this.nextBlink = t + 2.2 + Math.random() * 4.2; if (Math.random() < 0.18) this.nextBlink = t + 0.45; }
    var b = 0;
    if (this.blinkAt >= 0) { var u = (t - this.blinkAt) / 0.2; if (u >= 1) this.blinkAt = -1; else b = Math.sin(u * Math.PI); }
    this.p.blink = b;                       // the blink is its own fast curve
    var keys = ['jaw', 'wide', 'round', 'smile', 'brow', 'browIn', 'yaw', 'pitch', 'roll', 'breath'];
    for (var i = 0; i < keys.length; i++) follow(this, keys[i], dt, keys[i] === 'jaw' ? 38 : (keys[i] === 'yaw' || keys[i] === 'pitch' || keys[i] === 'roll' ? 6 : 14));
    this.render();
  };

  TalkingFace.prototype.render = function () {
    if (!this.gl) return;
    var m = this.m, P = this.p, base = this.base, pos = this.pos, n = m.pts.length;
    var fh = m.face_h, fw = m.face_w, cx = m.cx, mc = m.mouth_c, mw = m.mouth_w, pv = m.pivot;
    var jaw = clamp(P.jaw, 0, 1.1), wide = P.wide, round = P.round, smile = P.smile;
    var yaw = P.yaw * 0.045, pitch = P.pitch * 0.02, roll = P.roll * 0.012;
    var cr = Math.cos(roll), sr = Math.sin(roll);
    var jawDrop = fh * 0.1 * jaw;
    for (var i = 0; i < n; i++) {
      var x = base[i * 2], y = base[i * 2 + 1];
      // jaw
      y += jawDrop * m.w_jaw[i];
      x += (x < cx ? 1 : -1) * 0 ;
      // mouth shape: wide stretches sideways, round purses it
      var wl = m.w_lip[i];
      x = cx + (x - cx) * (1 + (wide * 0.14 - round * 0.28) * wl);
      y -= round * fh * 0.004 * wl;
      // smile: corners up and out, cheeks rise
      var wc = m.w_corner[i], side = x < cx ? -1 : 1;
      y -= smile * fh * 0.032 * wc; x += side * smile * mw * 0.06 * wc;
      y -= smile * fh * 0.01 * m.w_cheek[i];
      // brows
      y -= P.brow * fh * 0.04 * m.w_brow[i];
      y += P.browIn * fh * 0.012 * m.w_brow[i] * (Math.abs(x - cx) > fw * 0.15 ? 1 : -1.4);
      y -= P.browIn * fh * 0.018 * m.w_browin[i];
      x -= (x - cx) * P.browIn * 0.02 * m.w_browin[i];
      base_pos(pos, i, x, y);
    }
    // blink: the upper lids come down onto the lower lids
    var bl = P.blink;
    if (bl > 0.001) {
      blinkSide(m, pos, base, m.lid_r_up, m.lid_r_low, bl);
      blinkSide(m, pos, base, m.lid_l_up, m.lid_l_low, bl);
    }
    // the head: a small turn (parallax by depth), nod and tilt, around a pivot near the nose bridge
    var sc = 1 + P.breath * 0.004;
    for (var j = 0; j < n; j++) {
      var X = pos[j * 2] - pv[0], Y = pos[j * 2 + 1] - pv[1], z = m.z[j];
      var w = j < m.n ? 1 : 0.35;        // the background follows less, so the head moves against it
      X += yaw * z * fw * 6 * w + yaw * fw * 0.5 * w * 0.4;
      Y += pitch * fh * w + pitch * z * fw * 3 * w;
      var rx = X * cr - Y * sr, ry = X * sr + Y * cr;
      pos[j * 2] = pv[0] + rx * sc; pos[j * 2 + 1] = pv[1] + ry * sc;
    }
    var gl = this.gl;
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.posBuf);
    gl.bufferData(gl.ARRAY_BUFFER, pos, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(this.aP); gl.vertexAttribPointer(this.aP, 2, gl.FLOAT, false, 0, 0);
    gl.uniform1f(this.uWarm, 1.0);
    gl.drawElements(gl.TRIANGLES, this.nTri, gl.UNSIGNED_SHORT, 0);
    this.octx.clearRect(0, 0, this.over.width, this.over.height);
    this.mouth(pos);
    if (bl > 0.15) this.lashes(pos, bl);
  };

  function base_pos(a, i, x, y) { a[i * 2] = x; a[i * 2 + 1] = y; }
  function blinkSide(m, pos, base, up, low, bl) {
    for (var k = 0; k < up.length; k++) {
      var a = up[k], b = low[k];
      // pos was already built for this frame: pull the lid toward the lower lid
      var ya = pos[a * 2 + 1], yb = pos[b * 2 + 1];
      pos[a * 2 + 1] = ya + (yb - ya) * bl * 0.92;
    }
  }

  // a smooth closed curve through points (midpoint quadratics)
  function smooth(c, pts) {
    var n = pts.length, i;
    c.moveTo((pts[n - 1][0] + pts[0][0]) / 2, (pts[n - 1][1] + pts[0][1]) / 2);
    for (i = 0; i < n; i++) { var p = pts[i], q = pts[(i + 1) % n]; c.quadraticCurveTo(p[0], p[1], (p[0] + q[0]) / 2, (p[1] + q[1]) / 2); }
    c.closePath();
  }

  // the inside of the mouth, drawn over the face where the lips part: dark cavity, soft shadow under the
  // upper lip, a hint of tongue, and the teeth (upper, and the lower ones only when the mouth is wide)
  TalkingFace.prototype.mouth = function (pos) {
    var m = this.m, c = this.octx, W = this.over.width, k = W / m.w;
    var loop = m.inner, up = m.in_up, low = m.in_low, i;
    var ymin = 1e9, ymax = -1e9, xmin = 1e9, xmax = -1e9, pts = [];
    for (i = 0; i < loop.length; i++) {
      var x = pos[loop[i] * 2], y = pos[loop[i] * 2 + 1];
      pts.push([x, y]); ymin = Math.min(ymin, y); ymax = Math.max(ymax, y); xmin = Math.min(xmin, x); xmax = Math.max(xmax, x);
    }
    var open = ymax - ymin;
    if (open < 1.4) return;
    var cxm = (xmin + xmax) / 2, wd = xmax - xmin, fade = clamp((open - 1.4) / 3, 0, 1);
    c.save(); c.scale(k, k); c.globalAlpha = fade;
    c.beginPath(); smooth(c, pts);
    var gr = c.createLinearGradient(0, ymin, 0, ymax);
    gr.addColorStop(0, '#1d090d'); gr.addColorStop(0.5, '#3b1419'); gr.addColorStop(1, '#5b2329');
    c.fillStyle = gr; c.fill();
    c.save(); c.clip();
    // tongue
    var tg0 = c.createRadialGradient(cxm, ymax + open * 0.1, 1, cxm, ymax + open * 0.1, wd * 0.34);
    tg0.addColorStop(0, 'rgba(176,86,92,.85)'); tg0.addColorStop(1, 'rgba(120,48,56,0)');
    c.fillStyle = tg0; c.fillRect(xmin, ymin, wd, open * 1.4);
    // upper teeth: the middle of the upper inner lip, hanging a little way down
    var th = Math.min(open * 0.5, m.face_h * 0.024), a0 = 2, a1 = up.length - 3, top = [], bot = [];
    for (i = a0; i <= a1; i++) {
      var ux = pos[up[i] * 2], uy = pos[up[i] * 2 + 1], t = (i - a0) / (a1 - a0), hh = th * (0.55 + 0.45 * Math.sin(Math.PI * t));
      top.push([ux, uy - 0.6]); bot.push([ux, uy + hh]);
    }
    c.beginPath();
    c.moveTo(top[0][0], top[0][1]);
    for (i = 1; i < top.length; i++) c.lineTo(top[i][0], top[i][1]);
    for (i = bot.length - 1; i >= 0; i--) { var bq = bot[i], bn = bot[i - 1] || bq; c.quadraticCurveTo(bq[0], bq[1] + 0.8, (bq[0] + bn[0]) / 2, (bq[1] + bn[1]) / 2 + 0.4); }
    c.closePath();
    var tg = c.createLinearGradient(0, ymin, 0, ymin + th);
    tg.addColorStop(0, '#d9d2c7'); tg.addColorStop(0.35, '#f3eee6'); tg.addColorStop(1, '#d6cec2');
    c.fillStyle = tg; c.fill();
    // lower teeth, dimmer, only when open wide
    if (open > m.face_h * 0.065) {
      var lh = th * 0.5, l0 = 2, l1 = low.length - 3;
      c.beginPath();
      for (i = l0; i <= l1; i++) { var lx = pos[low[i] * 2], ly = pos[low[i] * 2 + 1]; if (i === l0) c.moveTo(lx, ly + 0.4); else c.lineTo(lx, ly + 0.4); }
      for (i = l1; i >= l0; i--) c.lineTo(pos[low[i] * 2], pos[low[i] * 2 + 1] - lh * Math.sin(Math.PI * (i - l0) / (l1 - l0)) );
      c.closePath(); c.fillStyle = 'rgba(214,206,194,.78)'; c.fill();
    }
    // shadow under the upper lip
    var sh = c.createLinearGradient(0, ymin, 0, ymin + open * 0.55);
    sh.addColorStop(0, 'rgba(10,2,4,.55)'); sh.addColorStop(1, 'rgba(10,2,4,0)');
    c.fillStyle = sh; c.fillRect(xmin, ymin, wd, open * 0.6);
    c.restore();
    // a soft edge, so the opening sits inside the lips
    c.beginPath(); smooth(c, pts); c.strokeStyle = 'rgba(70,24,30,.55)'; c.lineWidth = 1.0; c.stroke();
    c.restore();
  };

  // a fine lash line over a closed eye, so a blink reads as a lid, not a smear
  TalkingFace.prototype.lashes = function (pos, bl) {
    var m = this.m, c = this.octx, k = this.over.width / m.w;
    c.save(); c.scale(k, k);
    c.strokeStyle = 'rgba(40,24,20,' + (0.75 * bl) + ')'; c.lineWidth = 1.1; c.lineCap = 'round';
    [m.lid_r_up, m.lid_l_up].forEach(function (grp) {
      c.beginPath();
      grp.forEach(function (i, n) { var x = pos[i * 2], y = pos[i * 2 + 1] + 0.4; if (n === 0) c.moveTo(x, y); else c.lineTo(x, y); });
      c.stroke();
    });
    c.restore();
  };

  root.TalkingFace = TalkingFace;
})(window);
