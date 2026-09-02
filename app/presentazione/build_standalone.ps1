# build_standalone.ps1
# Assembla tutte le slide in un unico standalone.html apribile senza server

$BASE = "C:\Users\paolo\MCP_ECOM\app\presentazione"
$SLIDES_DIR = "$BASE\slides"
$INDEX = "$BASE\index.html"
$CSS = "$BASE\styles.css"
$OUTPUT = "$BASE\standalone.html"

# Leggi CSS
$css = Get-Content $CSS -Raw -Encoding UTF8

# Lista slide nell'ordine corretto
$slideFiles = @(
  'slide_01.html','slide_02.html','slide_03.html','slide_04.html',
  'slide_05.html','slide_06.html','slide_07.html','slide_08.html',
  'slide_09.html','slide_10.html','slide_11.html','slide_12.html',
  'slide_13.html','slide_14.html','slide_16.html','slide_17.html',
  'slide_18.html','slide_20.html','slide_21.html','slide_22.html',
  'slide_23.html','slide_24.html','slide_25.html'
)

$total = $slideFiles.Count

# Costruisci template tags
$templates = ""
for ($i = 0; $i -lt $slideFiles.Count; $i++) {
  $file = "$SLIDES_DIR\$($slideFiles[$i])"
  if (Test-Path $file) {
    $content = Get-Content $file -Raw -Encoding UTF8
    $templates += "`n<template id=`"slide-$i`">$content</template>"
  } else {
    Write-Warning "Slide not found: $file"
  }
}

# JavaScript standalone (no fetch)
$js = @"
const TOTAL = $total;
let current = -1;
const sections = [];

function execScripts(parent) {
  parent.querySelectorAll('script').forEach(old => {
    const s = document.createElement('script');
    Array.from(old.attributes).forEach(a => s.setAttribute(a.name, a.value));
    s.textContent = old.textContent;
    old.parentNode.replaceChild(s, old);
  });
}

function loadFromTemplate(index) {
  const tpl = document.getElementById('slide-' + index);
  if (!tpl) return null;
  const tmp = document.createElement('div');
  tmp.innerHTML = tpl.innerHTML.trim();
  const section = tmp.querySelector('section');
  if (!section) return null;
  section.dataset.idx = index;
  document.getElementById('deck').appendChild(section);
  execScripts(section);
  return section;
}

function scaleStage() {
  const deck = document.getElementById('deck');
  const s = Math.min(window.innerWidth / 1920, window.innerHeight / 1080);
  deck.style.transform = 'translate(-50%, -50%) scale(' + s + ')';
  deck.style.margin = '0';
}

function buildDots() {
  var c = document.getElementById('nav-dots');
  if (!c) return;
  for (var i = 0; i < TOTAL; i++) {
    (function(idx) {
      var d = document.createElement('span');
      d.className = 'dot';
      d.addEventListener('click', function() { showSlide(idx); });
      c.appendChild(d);
    })(i);
  }
}

function updateDots(index) {
  document.querySelectorAll('#nav-dots .dot').forEach(function(d, i) {
    d.classList.toggle('active', i === index);
  });
}

function updateSubtitle(sec) {
  var el = document.getElementById('slide-subtitle');
  if (!el || !sec) return;
  var spans = sec.querySelectorAll('.footer span');
  var text = '';
  for (var i = spans.length - 1; i >= 0; i--) {
    var t = (spans[i].textContent || '').trim();
    if (t && !/^\d/.test(t)) { text = t; break; }
  }
  el.textContent = text;
}

function showSlide(index) {
  if (index < 0 || index >= sections.length || index === current) return;
  var outSec = sections[current];
  var inSec  = sections[index];
  if (!inSec) return;

  var dir = index > current ? 1 : -1;

  if (outSec) gsap.killTweensOf(outSec);
  gsap.killTweensOf(inSec);

  if (outSec) outSec.style.display = 'block';

  inSec.style.display = 'block';
  void inSec.offsetWidth;
  inSec.classList.add('active');
  gsap.set(inSec, { opacity: 0, x: dir * 50 });

  if (outSec && outSec !== inSec) {
    gsap.to(outSec, {
      opacity: 0, x: -dir * 25, duration: 0.22, ease: 'power2.in',
      onComplete: function() {
        outSec.classList.remove('active');
        outSec.style.display = '';
        gsap.set(outSec, { clearProps: 'all' });
      }
    });
  }

  gsap.to(inSec, {
    opacity: 1, x: 0, duration: 0.30, ease: 'power2.out', delay: 0.05,
    onComplete: function() {
      gsap.set(inSec, { clearProps: 'opacity,x' });
      inSec.style.display = '';
    }
  });

  current = index;
  document.getElementById('progress-fill').style.width = ((current + 1) / TOTAL) * 100 + '%';
  history.replaceState(null, '', '#' + (current + 1));
  updateDots(current);
  updateSubtitle(inSec);
}

function next() { if (current < sections.length - 1) showSlide(current + 1); }
function prev() { if (current > 0) showSlide(current - 1); }

function init() {
  scaleStage();
  window.addEventListener('resize', scaleStage);
  buildDots();
  for (let i = 0; i < TOTAL; i++) {
    sections.push(loadFromTemplate(i));
  }
  const hash = parseInt(window.location.hash.replace('#', ''), 10);
  const start = (hash >= 1 && hash <= TOTAL) ? hash - 1 : 0;
  current = start;
  sections[start].classList.add('active');
  document.getElementById('progress-fill').style.width = ((start + 1) / TOTAL) * 100 + '%';
  history.replaceState(null, '', '#' + (start + 1));
  updateDots(start);
  updateSubtitle(sections[start]);
}

document.addEventListener('keydown', e => {
  if (e.key === 'ArrowRight' || e.key === ' ' || e.key === 'PageDown') { e.preventDefault(); next(); }
  if (e.key === 'ArrowLeft'  || e.key === 'PageUp')                    { e.preventDefault(); prev(); }
  if (e.key === 'f' || e.key === 'F') document.documentElement.requestFullscreen?.();
  if (e.key === 'Home') showSlide(0);
  if (e.key === 'End')  showSlide(sections.length - 1);
});

document.getElementById('nav-prev').addEventListener('click', prev);
document.getElementById('nav-next').addEventListener('click', next);

let touchX = 0;
document.addEventListener('touchstart', e => { touchX = e.touches[0].clientX; }, { passive: true });
document.addEventListener('touchend', e => {
  const dx = e.changedTouches[0].clientX - touchX;
  if (dx < -50) next();
  if (dx >  50) prev();
}, { passive: true });

init();
"@

# HTML completo
$html = @"
<!DOCTYPE html>
<html lang="it">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>MCP_ECOM — AI Shopping Assistant</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Instrument+Serif:ital,wght@0,400;1,400&family=JetBrains+Mono:wght@400;500;600&family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,1,0&display=swap" rel="stylesheet">
  <script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/gsap.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css" crossorigin="anonymous">
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js" crossorigin="anonymous"></script>
  <style>
$css
    html, body { width: 100%; height: 100%; margin: 0; padding: 0; }
    body { background: #0d0d0d; overflow: hidden; }
    #stage { position: fixed; inset: 0; }
    #deck { position: absolute; top: 50%; left: 50%; width: 1920px; height: 1080px; transform-origin: center center; transform: translate(-50%, -50%) scale(1); }
    #nav-dots { position:fixed; bottom:20px; left:50%; transform:translateX(-50%); display:flex; gap:7px; z-index:100; pointer-events:none; }
    #nav-dots .dot { width:5px; height:5px; border-radius:50%; background:rgba(147,147,159,0.35); cursor:pointer; pointer-events:all; transition:background 0.25s,transform 0.25s; }
    #nav-dots .dot.active { background:#0d7a5f; transform:scale(1.6); }
    #slide-subtitle { position:fixed; bottom:34px; left:50%; transform:translateX(-50%); font-family:'JetBrains Mono',monospace; font-size:10px; color:rgba(147,147,159,0.6); text-transform:uppercase; letter-spacing:0.2em; z-index:99; pointer-events:none; white-space:nowrap; }
    .logo { animation:logoPulse 3s ease-in-out infinite; }
    @keyframes logoPulse { 0%,100%{box-shadow:0 0 0 0 rgba(13,122,95,0.3);} 50%{box-shadow:0 0 0 6px rgba(13,122,95,0);} }
    .main[style*="flex-direction:column"] { justify-content:center !important; }
  </style>
</head>
<body>
  <div id="progress-bar"><div id="progress-fill"></div></div>
  <div id="stage"><div id="deck"></div></div>
  <div id="nav-prev">&#8592;</div>
  <div id="nav-next">&#8594;</div>

  <div id="nav-dots"></div>
  <div id="slide-subtitle"></div>

  <!-- Slide templates (no fetch needed) -->
  $templates

  <script>
$js
  </script>
</body>
</html>
"@

$html | Out-File -FilePath $OUTPUT -Encoding UTF8
Write-Host "✅ Standalone built: $OUTPUT" -ForegroundColor Green
Write-Host "   $($slideFiles.Count) slides embedded" -ForegroundColor Cyan

# Apri direttamente nel browser default
Start-Process $OUTPUT
