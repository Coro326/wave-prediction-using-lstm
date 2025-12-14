function predict(){
    fetch("/predict")
    .then(res => res.json())
    .then(data => {
        if (data.error) {
            document.getElementById("result").innerText = "Error: " + data.error + " - " + data.details;
        } else if (data.wave_height !== undefined && data.wave_height !== null) {
            document.getElementById("result").innerText =
                "Predicted Wave Height: " + data.wave_height + " meters";
        } else {
            document.getElementById("result").innerText = "Error: Invalid response from server";
        }
    })
    .catch(error => {
        document.getElementById("result").innerText = "Error: " + error.message;
    });
}
