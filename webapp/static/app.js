function updateClock() {
  const clock = document.getElementById("clock");
  if (!clock) return;

  const now = new Date();

  const date = now.toLocaleDateString("en-US", {
    year: "numeric",
    month: "long",
    day: "2-digit"
  });

  const time = now.toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false
  });

  clock.textContent = `${date} | ${time}`;
}

setInterval(updateClock, 1000);
updateClock();

async function updateStatus() {
  const res = await fetch("/api/status");
  const data = await res.json();

  const alertBox = document.getElementById("alertBox");
  const statusBox = document.getElementById("statusBox");

  if (data.status === "ALERT") {
    alertBox.style.display = "block";
    statusBox.style.display = "none";
  } else {
    alertBox.style.display = "none";
    statusBox.style.display = "block";
  }
}

async function updateLogs() {
  const res = await fetch("/api/logs");
  const logs = await res.json();

  const table = document.getElementById("logTable");
  table.innerHTML = "";

  logs.slice(0, 10).forEach((log, index) => {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${index + 1}</td>
      <td>${log.type}</td>
      <td>${log.confidence}</td>
      <td>${log.timestamp}</td>
    `;
    table.appendChild(row);
  });
}

setInterval(() => {
  updateStatus();
  updateLogs();
}, 500);
