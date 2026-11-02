/*
 * PhisScan front-end behaviour.
 *
 * Everything here is progressive enhancement: every form and link still
 * works with JavaScript turned off, because the server does all the real
 * validation. The site's Content-Security-Policy forbids inline scripts,
 * so all behaviour is attached from this file instead of onclick=""
 * attributes in the templates.
 */
(function () {
  "use strict";

  // Consent page: keep "I Consent and Continue" disabled until every box
  // is ticked, with a counter so people can see what's left. The server
  // checks this again, so this is purely to help the participant.
  var consentForm = document.getElementById("consent-form");
  if (consentForm) {
    var boxes = consentForm.querySelectorAll('input[type="checkbox"][data-required]');
    var consentButton = document.getElementById("consent-submit");
    var counter = document.getElementById("consent-count");

    var refreshConsentButton = function () {
      var ticked = 0;
      boxes.forEach(function (box) { if (box.checked) ticked++; });
      consentButton.disabled = ticked !== boxes.length;
      if (counter) counter.textContent = ticked + " of " + boxes.length + " statements confirmed";
    };
    boxes.forEach(function (box) { box.addEventListener("change", refreshConsentButton); });
    refreshConsentButton();
  }

  // Dashboard tabs. Without JS each tab is a normal link (?tab=url). With
  // JS we switch panels in place and update the address bar, so a refresh
  // or the back button still lands on the right tab.
  var showTab = function (name) {
    document.querySelectorAll(".tabs .tab").forEach(function (tab) {
      var active = tab.getAttribute("data-tab") === name;
      tab.classList.toggle("active", active);
      tab.setAttribute("aria-selected", active ? "true" : "false");
    });
    document.querySelectorAll(".tab-panel").forEach(function (panel) {
      panel.classList.toggle("active", panel.id === "panel-" + name);
    });
    document.querySelectorAll(".nav a[data-tab]").forEach(function (link) {
      link.classList.toggle("active", link.getAttribute("data-tab") === name);
    });
    if (window.history && history.replaceState) history.replaceState(null, "", "?tab=" + name);
  };

  document.querySelectorAll(".tabs .tab[data-tab]").forEach(function (tab) {
    tab.addEventListener("click", function (event) {
      event.preventDefault();
      showTab(tab.getAttribute("data-tab"));
    });
  });

  // The header links (URL Detection / Email Detection) point at the same
  // tabs. On the dashboard itself, switch in place rather than reloading.
  document.querySelectorAll(".nav a[data-tab]").forEach(function (link) {
    link.addEventListener("click", function (event) {
      if (document.querySelector(".tabs")) {
        event.preventDefault();
        showTab(link.getAttribute("data-tab"));
      }
    });
  });

  // "Simulated example" buttons only copy text into the form. Participants
  // can try the scanner without finding (or clicking) a real phishing link.
  document.querySelectorAll("[data-example-url]").forEach(function (button) {
    button.addEventListener("click", function () {
      var input = document.getElementById("url-input");
      input.value = button.getAttribute("data-example-url");
      input.focus();
    });
  });

  document.querySelectorAll("[data-example-email]").forEach(function (button) {
    button.addEventListener("click", function () {
      // The example emails are kept in hidden <textarea>s so their HTML
      // stays as inert text and is never parsed as markup.
      var example = document.getElementById(button.getAttribute("data-example-email"));
      if (!example) return;
      document.getElementById("email-subject").value = example.getAttribute("data-subject") || "";
      document.getElementById("email-sender").value = example.getAttribute("data-sender") || "";
      document.getElementById("email-body").value = example.textContent.trim();
    });
  });

  // Disable submit buttons after the first click to avoid double scans
  // and double form submissions. The setTimeout lets the browser
  // include the button in the submission before it's disabled.
  document.querySelectorAll("form[data-once]").forEach(function (form) {
    form.addEventListener("submit", function () {
      var button = form.querySelector('button[type="submit"]');
      if (!button) return;
      setTimeout(function () {
        button.disabled = true;
        button.textContent = button.getAttribute("data-busy") || "Please wait…";
      }, 0);
    });
  });

  // Admin: ask before deleting anything. Deleting participant data can't be undone.
  document.querySelectorAll("form[data-confirm]").forEach(function (form) {
    form.addEventListener("submit", function (event) {
      if (!window.confirm(form.getAttribute("data-confirm"))) event.preventDefault();
    });
  });
})();
