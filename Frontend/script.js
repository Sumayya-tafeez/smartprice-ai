async function loadPrices(scenario) {
  const res = await fetch('http://localhost:8000/optimize', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ scenario })
  });
  const data = await res.json();

  document.getElementById('uplift').innerHTML = `Average uplift: <strong>${data.summary.avg_uplift}%</strong>`;

  const container = document.getElementById('products');
  container.innerHTML = '';

  data.products.forEach(p => {
    const card = document.createElement('div');
    card.className = 'card';
    card.innerHTML = `
      <img src="https://source.unsplash.com/random/300x200/?${p['Product Name']}" alt="${p['Product Name']}">
      <h3>${p['Product Name'] || p['Product ID']}</h3>
      <p class="old-price">$${p['Current Price']}</p>
      <p class="new-price">$${p['Optimal Price']}</p>
      <div class="uplift-tag">+${p['Revenue Uplift %']}% uplift</div>
      <p><small>${p['Recommendation'] || 'Hold'}</small></p>
    `;
    container.appendChild(card);
  });
}

// Load default (Black Friday)
loadPrices('blackfriday');