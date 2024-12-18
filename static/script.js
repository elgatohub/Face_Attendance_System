// document.getElementById('take-attendance').addEventListener('click', function() {
//     runPythonScript(); // Run Python script
// });

// function runPythonScript() {
//     fetch('/run-script', {
//         method: 'POST'
//     })
//     .then(response => response.text())
//     .then(data => {
//         // Display the webcam feed received from the Python script
//         document.getElementById('webcam-container').innerHTML = data;
//     })
//     .catch(error => {
//         console.error('Error running Python script:', error);
//     });
// }

// document.getElementById('capture').addEventListener('click', function() {
//     const userId = document.getElementById('user-id').value;
//     const userName = document.getElementById('user-name').value;

//     fetch('/run-script1', {
//         method: 'POST',
//         headers: {
//             'Content-Type': 'application/json',
//         },
//         body: JSON.stringify({
//             id: userId,
//             name: userName
//         })
//     })
//     .then(response => response.text())
//     .then(data => {
//         document.getElementById('webcam-container').innerHTML = data;
//     })
//     .catch(error => {
//         console.error('Error running Python script:', error);
//     });
// });





// document.getElementById('take-attendance').addEventListener('click', function() {
//     runPythonScript();
// });

// function runPythonScript() {
//     fetch('/run-script', {
//         method: 'POST'
//     })
//     .then(response => response.text())
//     .then(data => {
//         alert(data);  // Display stdout in a pop-up alert
//         // You can also display the webcam feed if needed
//         document.getElementById('webcam-container').innerHTML = data;
//     })
//     .catch(error => {
//         console.error('Error running Python script:', error);
//     });
// }

// document.getElementById('capture').addEventListener('click', function() {
//     const userId = document.getElementById('user-id').value;
//     const userName = document.getElementById('user-name').value;

//     fetch('/run-script1', {
//         method: 'POST',
//         headers: {
//             'Content-Type': 'application/json',
//         },
//         body: JSON.stringify({
//             id: userId,
//             name: userName
//         })
//     })
//     .then(response => response.text())
//     .then(data => {
//         alert(data);  // Display stdout in a pop-up alert
//         // You can also display the webcam feed if needed
//         document.getElementById('webcam-container').innerHTML = data;
//     })
//     .catch(error => {
//         console.error('Error running Python script:', error);
//     });
// });



document.getElementById('take-attendance').addEventListener('click', function() {
    runPythonScript();
});

function runPythonScript() {
    fetch('/run-script', {
        method: 'POST'
    })
    .then(response => response.text())
    .then(data => {
        alert(data);  // Display stdout in a pop-up alert
        document.getElementById('webcam-container').innerHTML = data;
    })
    .catch(error => {
        console.error('Error running Python script:', error);
    });
}

document.getElementById('capture').addEventListener('click', function() {
    const userId = document.getElementById('user-id').value;
    const userName = document.getElementById('user-name').value;

    console.log(`Sending ID: ${userId}, Name: ${userName}`);  // Debug print

    fetch('/run-script1', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            Id: userId,
            name: userName
        })
    })
    .then(response => response.text())
    .then(data => {
        alert(data);  // Display stdout in a pop-up alert
        document.getElementById('webcam-container').innerHTML = data;
    })
    .catch(error => {
        console.error('Error running Python script:', error);
    });
});

