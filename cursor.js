(function () {
  'use strict';

  if (window.matchMedia('(pointer: coarse)').matches) return;

  var canvas = document.createElement('canvas');
  canvas.style.cssText = 'position:fixed;inset:0;width:100%;height:100%;pointer-events:none;z-index:9999;mix-blend-mode:screen;';
  document.body.appendChild(canvas);

  var gl = canvas.getContext('webgl2');
  if (!gl) { canvas.remove(); return; }

  var w = 0, h = 0, simW = 0, simH = 0;

  // ── shaders ──────────────────────────────────────────────────────────────

  var VS = '#version 300 es\nin vec2 a_pos;out vec2 v_uv;\nvoid main(){gl_Position=vec4(a_pos,0.,1.);v_uv=a_pos*.5+.5;}';

  var SIM_FS = [
    '#version 300 es',
    'precision highp float;',
    'uniform sampler2D u_prev;',
    'uniform vec2 u_texelSize;',
    'uniform vec4 u_from;',
    'uniform vec4 u_to;',
    'uniform vec2 u_vel;',
    'uniform float u_pushStrength;',
    'uniform vec3 u_dissipations;',
    'in vec2 v_uv;out vec4 fragColor;',
    'vec2 sdSeg(vec2 p,vec2 a,vec2 b){vec2 pa=p-a,ba=b-a;float h=clamp(dot(pa,ba)/dot(ba,ba),0.,1.);return vec2(length(pa-ba*h),h);}',
    'void main(){',
      'vec2 res=sdSeg(gl_FragCoord.xy,u_from.xy,u_to.xy);',
      'vec2 rw=mix(u_from.zw,u_to.zw,res.y);',
      'float d=1.-smoothstep(-.01,rw.x,res.x);',
      'vec4 ld=texture(u_prev,v_uv);',
      'vec2 vi=(0.5-ld.xy)*u_pushStrength;',
      'vec4 data=texture(u_prev,v_uv+vi*u_texelSize);',
      'data.xy-=0.5;',
      'vec4 diss=vec4(u_dissipations.xxy,u_dissipations.z);',
      'vec4 delta=(diss-1.)*data;',
      'delta+=vec4(u_vel*d,rw.yy*d);',
      'delta.zw=sign(delta.zw)*max(vec2(0.004),abs(delta.zw));',
      'data+=delta;data.xy+=0.5;',
      'fragColor=clamp(data,vec4(0.),vec4(1.));',
    '}'
  ].join('\n');

  var DISP_FS = [
    '#version 300 es',
    'precision highp float;',
    'uniform sampler2D u_paint;',
    'uniform float u_rgbShift;',
    'uniform float u_colorMult;',
    'uniform float u_shade;',
    'in vec2 v_uv;out vec4 fragColor;',
    'void main(){',
      'vec4 data=texture(u_paint,v_uv);',
      'float weight=(data.z+data.w)*.5;',
      'vec2 vel=(0.5-data.xy-.001)*2.*weight;',
      'vec3 col=sin(vec3(vel.x+vel.y)*40.+vec3(0.,2.,4.)*u_rgbShift)',
        '*smoothstep(.4,-.9,weight)*u_shade*max(abs(vel.x),abs(vel.y))*u_colorMult;',
      'col=abs(col);',
      'fragColor=vec4(col,clamp(weight*3.,0.,1.));',
    '}'
  ].join('\n');

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

  var simProg  = mkProg(VS, SIM_FS);
  var dispProg = mkProg(VS, DISP_FS);

  var quadBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, 1,1]), gl.STATIC_DRAW);

  function bindQuad(prog) {
    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
    var loc = gl.getAttribLocation(prog, 'a_pos');
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
  }

  // ── FBO ping-pong (RGBA8 — no extension needed) ───────────────────────────

  var simPair = [null, null], simCurr = 0;

  function mkFBO(fw, fh) {
    var tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, fw, fh, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    var fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    gl.clearColor(0.502, 0.502, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    return { fbo: fbo, tex: tex };
  }

  function resize() {
    w = window.innerWidth;
    h = window.innerHeight;
    simW = Math.max(1, Math.ceil(w / 4));
    simH = Math.max(1, Math.ceil(h / 4));
    canvas.width  = w;
    canvas.height = h;
    if (simPair[0]) {
      gl.deleteTexture(simPair[0].tex); gl.deleteFramebuffer(simPair[0].fbo);
      gl.deleteTexture(simPair[1].tex); gl.deleteFramebuffer(simPair[1].fbo);
    }
    simPair[0] = mkFBO(simW, simH);
    simPair[1] = mkFBO(simW, simH);
    simCurr = 0;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  window.addEventListener('resize', resize);
  resize();

  // ── mouse ─────────────────────────────────────────────────────────────────

  var mx = w / 2, my = h / 2, pmx = w / 2, pmy = h / 2, mvx = 0, mvy = 0;

  window.addEventListener('mousemove', function (e) {
    pmx = mx; pmy = my;
    mx = e.clientX; my = e.clientY;
    mvx = mx - pmx; mvy = my - pmy;
  });

  // ── render ────────────────────────────────────────────────────────────────

  function tick() {
    requestAnimationFrame(tick);

    var readIdx  = simCurr;
    var writeIdx = 1 - simCurr;

    // simulation pass
    gl.bindFramebuffer(gl.FRAMEBUFFER, simPair[writeIdx].fbo);
    gl.viewport(0, 0, simW, simH);
    gl.useProgram(simProg);
    bindQuad(simProg);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, simPair[readIdx].tex);
    gl.uniform1i(gl.getUniformLocation(simProg, 'u_prev'), 0);
    gl.uniform2f(gl.getUniformLocation(simProg, 'u_texelSize'), 1 / simW, 1 / simH);
    gl.uniform1f(gl.getUniformLocation(simProg, 'u_pushStrength'), 10);
    gl.uniform3f(gl.getUniformLocation(simProg, 'u_dissipations'), 0.985, 0.985, 0.5);

    var fromX = pmx / w * simW, fromY = (1 - pmy / h) * simH;
    var toX   = mx  / w * simW, toY   = (1 - my  / h) * simH;
    gl.uniform4f(gl.getUniformLocation(simProg, 'u_from'), fromX, fromY, 18, 1);
    gl.uniform4f(gl.getUniformLocation(simProg, 'u_to'),   toX,   toY,   18, 1);

    var speed = Math.sqrt(mvx * mvx + mvy * mvy);
    var nx = speed > 0 ? mvx / speed : 0;
    var ny = speed > 0 ? mvy / speed : 0;
    var mag = Math.min(speed / Math.max(w, h) * 8, 0.08);
    gl.uniform2f(gl.getUniformLocation(simProg, 'u_vel'), nx * mag, -ny * mag);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    simCurr = writeIdx;

    // display pass
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, w, h);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(dispProg);
    bindQuad(dispProg);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, simPair[writeIdx].tex);
    gl.uniform1i(gl.getUniformLocation(dispProg, 'u_paint'), 0);
    gl.uniform1f(gl.getUniformLocation(dispProg, 'u_rgbShift'), 1.0);
    gl.uniform1f(gl.getUniformLocation(dispProg, 'u_colorMult'), 5.0);
    gl.uniform1f(gl.getUniformLocation(dispProg, 'u_shade'), 1.25);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  requestAnimationFrame(tick);
}());
