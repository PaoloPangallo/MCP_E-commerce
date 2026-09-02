const SLIDES = [
  'slides/slide_01.html',
  'slides/slide_02.html',
  'slides/slide_03.html',
  'slides/slide_04.html',
  'slides/slide_05.html',
  'slides/slide_06.html',
  'slides/slide_07.html',
  'slides/slide_08.html',
  'slides/slide_09.html',
  'slides/slide_10.html',
  'slides/slide_11.html',
  'slides/slide_12.html',
  'slides/slide_13.html',
  'slides/slide_14.html',
  'slides/slide_16.html',
  'slides/slide_17.html',
  'slides/slide_18.html',
  'slides/slide_20.html',
  'slides/slide_21.html',
  'slides/slide_22.html',
  'slides/slide_23.html',
  'slides/slide_24.html',
  'slides/slide_25.html',
];

const TOTAL = SLIDES.length;
let current = 0;
const sections = [];

function execScripts(parent) {
  parent.querySelectorAll('script').forEach(old => {
    const s = document.createElement('script');
    Array.from(old.attributes).forEach(a => s.setAttribute(a.name, a.value));
    s.textContent = old.textContent;
    old.parentNode.replaceChild(s, old);
  });
}

async function loadSlide(url, index) {
  try {
    const html = SLIDES_DATA[url];
    if (!html) {
      console.warn('Slide not found in preloaded data:', url);
      return null;
    }
    const tmp = document.createElement('div');
    tmp.innerHTML = html.trim();
    const section = tmp.querySelector('section');
    if (!section) return null;
    section.dataset.idx = index;
    document.getElementById('deck').appendChild(section);
    execScripts(section);
    return section;
  } catch (e) {
    console.warn('Failed to load', url, e);
    return null;
  }
}

function scaleStage() {
  const deck = document.getElementById('deck');
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  const s = Math.min(vw / 1920, vh / 1080);
  deck.style.transform = `translate(-50%, -50%) scale(${s})`;
}

function showSlide(index) {
  if (index < 0 || index >= sections.length) return;
  sections.forEach((s, i) => {
    if (!s) return;
    const wasActive = s.classList.contains('active');
    const willActive = i === index;
    if (!wasActive && willActive) {
      // Reset animations by removing and re-adding active
      s.classList.remove('active');
      void s.offsetWidth; // force reflow
    }
    s.classList.toggle('active', willActive);
  });
  current = index;
  const pct = ((current + 1) / TOTAL) * 100;
  document.getElementById('progress-fill').style.width = pct + '%';
  // Update URL hash for bookmarking
  history.replaceState(null, '', '#' + (current + 1));
}

function next() { if (current < sections.length - 1) showSlide(current + 1); }
function prev() { if (current > 0) showSlide(current - 1); }

async function init() {
  scaleStage();
  window.addEventListener('resize', scaleStage);

  // Load all slides in order
  for (let i = 0; i < SLIDES.length; i++) {
    const section = await loadSlide(SLIDES[i], i);
    sections.push(section);
  }

  // Start from hash or 0
  const hash = parseInt(window.location.hash.replace('#', ''), 10);
  const startIndex = (hash >= 1 && hash <= TOTAL) ? hash - 1 : 0;
  showSlide(startIndex);
}

// Keyboard nav
document.addEventListener('keydown', e => {
  if (e.key === 'ArrowRight' || e.key === ' ' || e.key === 'PageDown') { e.preventDefault(); next(); }
  if (e.key === 'ArrowLeft'  || e.key === 'PageUp')                    { e.preventDefault(); prev(); }
  if (e.key === 'f' || e.key === 'F') document.documentElement.requestFullscreen?.();
  if (e.key === 'Home') showSlide(0);
  if (e.key === 'End')  showSlide(sections.length - 1);
});

// Click nav
document.getElementById('nav-prev').addEventListener('click', prev);
document.getElementById('nav-next').addEventListener('click', next);

// Touch swipe
let touchX = 0;
document.addEventListener('touchstart', e => { touchX = e.touches[0].clientX; }, { passive: true });
document.addEventListener('touchend', e => {
  const dx = e.changedTouches[0].clientX - touchX;
  if (dx < -50) next();
  if (dx >  50) prev();
}, { passive: true });

init();
