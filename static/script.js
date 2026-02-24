document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const submitBtn = document.getElementById('submitBtn');
    const btnText = submitBtn.querySelector('.btn-text');
    const loader = document.getElementById('loader');

    // Show loading state
    btnText.style.display = 'none';
    loader.style.display = 'block';
    submitBtn.disabled = true;

    // Collect data
    const formData = {
        gender: document.getElementById('gender').value,
        age: document.getElementById('age').value,
        hemoglobin: document.getElementById('hemoglobin').value,
        rbc: document.getElementById('rbc').value,
        mcv: document.getElementById('mcv').value,
        mch: document.getElementById('mch').value,
        mchc: document.getElementById('mchc').value,
        hematocrit: document.getElementById('hematocrit').value
    };

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });

        if (!response.ok) {
            throw new Error('Prediction failed');
        }

        const result = await response.json();
        showResult(result);
    } catch (error) {
        alert('An error occurred: ' + error.message);
    } finally {
        // Reset loading state
        btnText.style.display = 'block';
        loader.style.display = 'none';
        submitBtn.disabled = false;
    }
});

function showResult(result) {
    const modal = document.getElementById('resultModal');
    const statusH2 = document.getElementById('anemiaStatus');
    const probabilityP = document.getElementById('probabilityText');
    const indicator = document.getElementById('statusIndicator');

    statusH2.innerText = result.Anemia === 'Yes' ? 'Anemia Detected' : 'No Anemia Detected';
    probabilityP.innerText = `Probability Score: ${result.Probability}`;
    
    indicator.className = 'status-indicator';
    if (result.Anemia === 'Yes') {
        indicator.classList.add('status-yes');
        indicator.innerHTML = '⚠️';
    } else {
        indicator.classList.add('status-no');
        indicator.innerHTML = '✅';
    }

    modal.style.display = 'flex';
}

function closeResult() {
    document.getElementById('resultModal').style.display = 'none';
}
