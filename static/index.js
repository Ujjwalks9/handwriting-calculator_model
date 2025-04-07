const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

let drawing = false;

ctx.lineWidth = 10;
ctx.lineCap = 'round';
ctx.strokeStyle = '#000';

canvas.addEventListener('mousedown', (e) => {
  drawing = true;
  ctx.beginPath();
  ctx.moveTo(e.offsetX, e.offsetY);
});

canvas.addEventListener('mousemove', (e) => {
  if (drawing) {
    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.stroke();
  }
});

canvas.addEventListener('mouseup', () => {
  drawing = false;
});

canvas.addEventListener('mouseleave', () => {
  drawing = false;
});

function clearCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  document.getElementById('expression').textContent = '';
  document.getElementById('result').textContent = '';
}

function submitCanvas() {
  const imageData = canvas.toDataURL('image/png');

  fetch('/predict', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ image: imageData })
  })
    .then(response => response.json())
    .then(data => {
      document.getElementById('expression').textContent = data.expression;
      document.getElementById('result').textContent = data.result;
    })
    .catch(error => {
      console.error('Error:', error);
    });
}
