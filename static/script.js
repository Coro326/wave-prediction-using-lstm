let waveChart = null;

function predict() {
    fetch("/predict")
        .then((res) => res.json())
        .then((data) => {
            const resultEl = document.getElementById("result");

            if (data.error) {
                resultEl.innerText = "Error: " + data.error + " - " + data.details;
                return;
            }

            if (data.wave_height === undefined || data.wave_height === null) {
                resultEl.innerText = "Error: Invalid response from server";
                return;
            }

            // Show numeric prediction
            resultEl.innerText =
                "Predicted Wave Height: " + data.wave_height + " meters";

            // Show the last 10 values used for prediction (if provided)
            const recent = Array.isArray(data.recent_wave_heights)
                ? data.recent_wave_heights
                : [];
            const listEl = document.getElementById("recent-list");
            if (listEl) {
                listEl.innerHTML = "";
                recent.forEach((v, idx) => {
                    const li = document.createElement("li");
                    li.textContent = `Step ${idx + 1}: ${v.toFixed(3)} m`;
                    listEl.appendChild(li);
                });
            }

            // Build/update the chart with recent values + predicted next point
            const ctx = document.getElementById("waveChart");
            if (ctx && recent.length > 0) {
                const labels = [];
                const values = [];

                for (let i = 0; i < recent.length; i++) {
                    labels.push("Obs " + (i + 1));
                    values.push(recent[i]);
                }
                labels.push("Predicted");
                values.push(data.wave_height);

                if (waveChart) {
                    waveChart.data.labels = labels;
                    waveChart.data.datasets[0].data = values;
                    waveChart.update();
                } else {
                    waveChart = new Chart(ctx, {
                        type: "line",
                        data: {
                            labels: labels,
                            datasets: [
                                {
                                    label: "Wave Height (m)",
                                    data: values,
                                    borderColor: "rgba(54, 162, 235, 1)",
                                    backgroundColor: "rgba(54, 162, 235, 0.2)",
                                    tension: 0.2,
                                },
                            ],
                        },
                        options: {
                            responsive: true,
                            scales: {
                                y: {
                                    beginAtZero: true,
                                    title: {
                                        display: true,
                                        text: "Wave Height (m)",
                                    },
                                },
                                x: {
                                    title: {
                                        display: true,
                                        text: "Time Steps (recent → predicted)",
                                    },
                                },
                            },
                        },
                    });
                }
            }
        })
        .catch((error) => {
            document.getElementById("result").innerText =
                "Error: " + error.message;
        });
}
