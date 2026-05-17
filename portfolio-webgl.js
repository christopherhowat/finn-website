(function () {
  'use strict';

  // ── GL setup ──────────────────────────────────────────────────────────────

  var stage  = document.querySelector('.portfolio-stage');
  var canvas = document.createElement('canvas');
  canvas.style.cssText = 'position:absolute;inset:0;';
  stage.insertBefore(canvas, stage.firstChild);

  var gl = canvas.getContext('webgl2');
  if (!gl) { canvas.remove(); return; }

  // Hide the CSS-rendered frames — WebGL takes over
  var framesEl = document.getElementById('portfolioFrames');
  if (framesEl) framesEl.style.visibility = 'hidden';

  // ── helpers ───────────────────────────────────────────────────────────────

  function mkShader(type, src) {
    var s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS))
      console.error('shader:', gl.getShaderInfoLog(s));
    return s;
  }
  function mkProg(vs, fs) {
    var p = gl.createProgram();
    gl.attachShader(p, mkShader(gl.VERTEX_SHADER, vs));
    gl.attachShader(p, mkShader(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS))
      console.error('program:', gl.getProgramInfoLog(p));
    return p;
  }
  function mkFBO(fw, fh, linear) {
    var tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, fw, fh, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    var f = linear ? gl.LINEAR : gl.NEAREST;
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, f);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, f);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    var fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    return { fbo: fbo, tex: tex };
  }

  // ── shaders ───────────────────────────────────────────────────────────────

  // Fullscreen quad (for sim and distort passes)
  var VS_QUAD = '#version 300 es\nin vec2 a_pos;out vec2 v_uv;\nvoid main(){gl_Position=vec4(a_pos,0.,1.);v_uv=a_pos*.5+.5;}';

  // Positioned photo quad with object-fit:cover + border-radius SDF
  var VS_PHOTO = [
    '#version 300 es',
    'uniform vec2 u_center;',  // clip-space center
    'uniform vec2 u_half;',    // clip-space half-extents
    'in vec2 a_pos;out vec2 v_uv;',
    'void main(){',
      'gl_Position=vec4(a_pos*u_half+u_center,0.,1.);',
      'v_uv=vec2(a_pos.x*.5+.5,-.5*a_pos.y+.5);', // Y-flip for image orientation
    '}'
  ].join('\n');

  var FS_PHOTO = [
    '#version 300 es',
    'precision highp float;',
    'uniform sampler2D u_photo;',
    'uniform float u_frameAR;',  // frame width/height
    'uniform float u_photoAR;',  // photo natural width/height
    'uniform vec2 u_frameSize;', // frame size in pixels (for border-radius SDF)
    'uniform float u_radius;',   // border-radius in pixels
    'in vec2 v_uv;out vec4 fragColor;',
    'void main(){',
      // border-radius SDF
      'vec2 px=v_uv*u_frameSize;',
      'vec2 q=abs(px-u_frameSize*.5)-u_frameSize*.5+u_radius;',
      'float d=length(max(q,0.))+min(max(q.x,q.y),0.)-u_radius;',
      'if(d>0.5)discard;',
      // object-fit: cover
      'vec2 uv=v_uv;',
      'if(u_photoAR>u_frameAR){float sx=u_frameAR/u_photoAR;uv.x=(uv.x-.5)/sx+.5;}',
      'else{float sy=u_photoAR/u_frameAR;uv.y=(uv.y-.5)/sy+.5;}',
      'fragColor=texture(u_photo,clamp(uv,0.,1.));',
    '}'
  ].join('\n');

  // Fluid simulation (lusion's frag$n)
  var FS_SIM = [
    '#version 300 es',
    'precision highp float;',
    'uniform sampler2D u_prev;',
    'uniform vec2 u_texelSize;',
    'uniform vec4 u_from;',
    'uniform vec4 u_to;',
    'uniform vec2 u_vel;',
    'uniform float u_pushStr;',
    'uniform vec3 u_diss;',
    'in vec2 v_uv;out vec4 fragColor;',
    'vec2 sdSeg(vec2 p,vec2 a,vec2 b){vec2 pa=p-a,ba=b-a;float h=clamp(dot(pa,ba)/dot(ba,ba),0.,1.);return vec2(length(pa-ba*h),h);}',
    'void main(){',
      'vec2 res=sdSeg(gl_FragCoord.xy,u_from.xy,u_to.xy);',
      'vec2 rw=mix(u_from.zw,u_to.zw,res.y);',
      'float d=1.-smoothstep(-.01,rw.x,res.x);',
      'vec4 ld=texture(u_prev,v_uv);',
      'vec2 vi=(0.5-ld.xy)*u_pushStr;',
      'vec4 data=texture(u_prev,v_uv+vi*u_texelSize);',
      'data.xy-=0.5;',
      'vec4 delta=(vec4(u_diss.xxy,u_diss.z)-1.)*data;',
      'delta+=vec4(u_vel*d,rw.yy*d);',
      'delta.zw=sign(delta.zw)*max(vec2(0.004),abs(delta.zw));',
      'data+=delta;data.xy+=0.5;',
      'fragColor=clamp(data,vec4(0.),vec4(1.));',
    '}'
  ].join('\n');

  // Distortion pass (lusion's frag$1 — samples scene FBO through velocity field)
  var FS_DISTORT = [
    '#version 300 es',
    'precision highp float;',
    'uniform sampler2D u_scene;',
    'uniform sampler2D u_paint;',
    'uniform vec2 u_paintTexelSize;',
    'uniform float u_amount;',
    'uniform float u_rgbShift;',
    'uniform float u_multiplier;',
    'uniform float u_colorMult;',
    'uniform float u_shade;',
    'in vec2 v_uv;out vec4 fragColor;',
    'void main(){',
      'vec4 data=texture(u_paint,v_uv);',
      'float weight=(data.z+data.w)*.5;',
      'vec2 vel=(0.5-data.xy-.001)*2.*weight;',
      'vec2 velocity=vel*u_amount/4.*u_paintTexelSize*u_multiplier;',
      'vec2 uv=v_uv;',
      'vec4 color=vec4(0.);',
      'for(int i=0;i<9;i++){color+=texture(u_scene,uv);uv+=velocity;}',
      'color/=9.;',
      'color.rgb+=sin(vec3(vel.x+vel.y)*40.+vec3(0.,2.,4.)*u_rgbShift)',
        '*smoothstep(.4,-.9,weight)*u_shade*max(abs(vel.x),abs(vel.y))*u_colorMult;',
      'fragColor=color;',
    '}'
  ].join('\n');

  var photoProg   = mkProg(VS_PHOTO, FS_PHOTO);
  var simProg     = mkProg(VS_QUAD,  FS_SIM);
  var distortProg = mkProg(VS_QUAD,  FS_DISTORT);

  // Shared fullscreen quad
  var quadBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, 1,1]), gl.STATIC_DRAW);

  function bindQuad(prog) {
    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
    var loc = gl.getAttribLocation(prog, 'a_pos');
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
  }

  // Uniform location cache
  var uLoc = {};
  function u(prog, name) {
    var key = prog + '|' + name;
    if (!(key in uLoc)) uLoc[key] = gl.getUniformLocation(prog, name);
    return uLoc[key];
  }

  // ── photos ────────────────────────────────────────────────────────────────

  var imgEls = Array.from(document.querySelectorAll('.portfolio-frame__figure img'));
  var N = imgEls.length;
  var photoTex = new Array(N).fill(null);
  var photoAR  = new Array(N).fill(1.0);

  imgEls.forEach(function (el, i) {
    var img = new Image();
    img.onload = function () {
      photoAR[i] = img.naturalWidth / img.naturalHeight;
      var tex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.generateMipmap(gl.TEXTURE_2D);
      photoTex[i] = tex;
    };
    img.src = el.getAttribute('src');
  });

  // ── sizing ────────────────────────────────────────────────────────────────

  var w = 0, h = 0, simW = 0, simH = 0;
  var sceneFBO = null, simPair = [null, null], simCurr = 0;

  function resize() {
    w = window.innerWidth;
    h = window.innerHeight;
    simW = Math.max(1, Math.ceil(w / 4));
    simH = Math.max(1, Math.ceil(h / 4));
    canvas.width  = w;
    canvas.height = h;

    if (sceneFBO)    { gl.deleteTexture(sceneFBO.tex);    gl.deleteFramebuffer(sceneFBO.fbo);    }
    if (simPair[0])  { gl.deleteTexture(simPair[0].tex);  gl.deleteFramebuffer(simPair[0].fbo);  }
    if (simPair[1])  { gl.deleteTexture(simPair[1].tex);  gl.deleteFramebuffer(simPair[1].fbo);  }

    sceneFBO   = mkFBO(w, h, true);
    simPair[0] = mkFBO(simW, simH, true);
    simPair[1] = mkFBO(simW, simH, true);

    // Init sim textures to (0.5, 0.5, 0, 0) = zero velocity
    [simPair[0], simPair[1]].forEach(function (s) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, s.fbo);
      gl.viewport(0, 0, simW, simH);
      gl.clearColor(0.502, 0.502, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
    });

    simCurr = 0;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    uLoc = {}; // clear cache since FBOs changed
  }

  window.addEventListener('resize', function () { resize(); });
  resize();

  // ── mouse ─────────────────────────────────────────────────────────────────

  var isCoarse = window.matchMedia('(pointer: coarse)').matches;
  var mx = w / 2, my = h / 2, pmx = w / 2, pmy = h / 2;
  // Accumulated velocity with momentum (lusion: vel *= 0.8 + delta * dt * 0.8)
  var velX = 0, velY = 0;
  var lastTickTime = performance.now();

  if (!isCoarse) {
    window.addEventListener('mousemove', function (e) {
      pmx = mx; pmy = my;
      mx = e.clientX; my = e.clientY;
    });
  }

  // ── page background colour (CSS --color-background ≈ hsl(60 28% 96%)) ────

  var BG = [0.98, 0.976, 0.953];

  // ── render passes ─────────────────────────────────────────────────────────

  function renderScene(pfState) {
    var s      = pfState.pos    || 0;
    var scaleD = pfState.scaleD || 1;
    var step   = pfState.step   || 1;
    var mL     = pfState.marginLeft || 0;
    var frameW = pfState.frameW || w * 0.65;
    var frameH = pfState.frameH || h * 0.75;
    var fAR    = frameW / frameH;

    gl.bindFramebuffer(gl.FRAMEBUFFER, sceneFBO.fbo);
    gl.viewport(0, 0, w, h);
    gl.clearColor(BG[0], BG[1], BG[2], 1);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.useProgram(photoProg);
    bindQuad(photoProg);
    gl.disable(gl.BLEND);

    for (var i = 0; i < N; i++) {
      if (!photoTex[i]) continue;

      // Convert CSS layout to WebGL clip space
      // CSS: translate3d(-s,0,0) scale(scaleD) applied to .portfolio-frames
      //      transform-origin: 50% 50% of viewport (position:absolute;inset:0)
      // Order: first scale around viewport center, then translate -s
      var xCenter = mL + i * step + frameW / 2;
      var xFinal  = (xCenter - w / 2) * scaleD + w / 2 - s;
      var yFinal  = h / 2;

      // Cull off-screen frames
      var halfWpx = frameW * scaleD / 2;
      if (xFinal + halfWpx < 0 || xFinal - halfWpx > w) continue;

      var clipX = xFinal / (w / 2) - 1;
      var clipY = 0; // yFinal = h/2 always maps to clip 0

      var halfWclip = halfWpx / (w / 2);
      var halfHclip = (frameH * scaleD / 2) / (h / 2);

      gl.uniform2f(u(photoProg, 'u_center'), clipX, clipY);
      gl.uniform2f(u(photoProg, 'u_half'),   halfWclip, halfHclip);
      gl.uniform1f(u(photoProg, 'u_frameAR'), fAR);
      gl.uniform1f(u(photoProg, 'u_photoAR'), photoAR[i]);
      gl.uniform2f(u(photoProg, 'u_frameSize'), frameW * scaleD, frameH * scaleD);
      gl.uniform1f(u(photoProg, 'u_radius'), 8);

      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, photoTex[i]);
      gl.uniform1i(u(photoProg, 'u_photo'), 0);

      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }
  }

  function stepSim() {
    var now = performance.now();
    var dt  = Math.min((now - lastTickTime) / 1000, 0.05);
    lastTickTime = now;

    // Lusion: vel = vel * 0.8 + delta * dt * 0.8  (accelerationDissipation = 0.8)
    var dx = (mx - pmx) / Math.max(w, h);
    var dy = (my - pmy) / Math.max(w, h);
    velX = velX * 0.8 + dx * dt * 0.8;
    velY = velY * 0.8 + dy * dt * 0.8;

    // Lusion radius: min(pixelDist, 100) / viewportH * simH
    var pixelDist = Math.sqrt((mx - pmx) * (mx - pmx) + (my - pmy) * (my - pmy));
    var radius = Math.min(pixelDist, 100) / h * simH;

    pmx = mx; pmy = my;

    var readIdx  = simCurr;
    var writeIdx = 1 - simCurr;

    gl.bindFramebuffer(gl.FRAMEBUFFER, simPair[writeIdx].fbo);
    gl.viewport(0, 0, simW, simH);

    gl.useProgram(simProg);
    bindQuad(simProg);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, simPair[readIdx].tex);
    gl.uniform1i(u(simProg, 'u_prev'), 0);
    gl.uniform2f(u(simProg, 'u_texelSize'), 1 / simW, 1 / simH);
    gl.uniform1f(u(simProg, 'u_pushStr'), 25);
    gl.uniform3f(u(simProg, 'u_diss'), 0.985, 0.985, 0.5);

    var fromX = pmx / w * simW, fromY = (1 - pmy / h) * simH;
    var toX   = mx  / w * simW, toY   = (1 - my  / h) * simH;
    gl.uniform4f(u(simProg, 'u_from'), fromX, fromY, radius, 1);
    gl.uniform4f(u(simProg, 'u_to'),   toX,   toY,   radius, 1);

    gl.uniform2f(u(simProg, 'u_vel'), velX, -velY);

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    simCurr = writeIdx;
  }

  function distortDisplay() {
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, w, h);

    gl.useProgram(distortProg);
    bindQuad(distortProg);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, sceneFBO.tex);
    gl.uniform1i(u(distortProg, 'u_scene'), 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, simPair[simCurr].tex);
    gl.uniform1i(u(distortProg, 'u_paint'), 1);

    gl.uniform2f(u(distortProg, 'u_paintTexelSize'), 1 / simW, 1 / simH);
    gl.uniform1f(u(distortProg, 'u_amount'),    20);
    gl.uniform1f(u(distortProg, 'u_rgbShift'),  1.0);
    gl.uniform1f(u(distortProg, 'u_multiplier'), 1.25);
    gl.uniform1f(u(distortProg, 'u_colorMult'), 1.0);
    gl.uniform1f(u(distortProg, 'u_shade'),     1.25);

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  // ── main loop ─────────────────────────────────────────────────────────────

  function tick() {
    requestAnimationFrame(tick);
    var pf = window._pfState || { pos: 0, scaleD: 1 };
    renderScene(pf);
    stepSim();
    distortDisplay();
  }

  requestAnimationFrame(tick);
}());
