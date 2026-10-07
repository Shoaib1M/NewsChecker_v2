/*
FILE PURPOSE:
The main navigation bar at the top of the screen.

FLOW:
1. Renders the logo on the left.
2. Renders navigation links in the middle.
3. Renders the History button and the Google Sign-In (or user profile) on the right.

WHY THIS EXISTS:
Provides consistent navigation across all pages of the app. It also manages the entry point
for user authentication by providing a container (`#google-signin-btn`) for the Google API to render into.
*/

import { CheckCircle2, X } from "lucide-react";

export default function Header({ user, onSignOut, onHistoryToggle, historyCount, currentPage, onNavigate }) {
  // Define our pages here so we can loop over them below to create the menu.
  const navItems = [
    { key: "", label: "Check" },
    { key: "comparison", label: "Comparison" },
    { key: "how-it-works", label: "How It Works" },
  ];

  return (
    <header className="header">
      <div className="header-inner">
        {/* Branding / Logo */}
        <a href="#/" className="header-brand" onClick={(e) => { e.preventDefault(); onNavigate(""); }}>
          <span className="header-mark"><CheckCircle2 size={18} /></span>
          <span className="header-title">newschecker</span>
        </a>

        {/* Center Navigation Links */}
        <nav className="header-nav">
          {navItems.map((item) => (
            <a
              key={item.key}
              href={`#/${item.key}`}
              className={`header-nav-link ${currentPage === item.key ? "active" : ""}`}
              onClick={(e) => {
                e.preventDefault();
                onNavigate(item.key);
              }}
            >
              {item.label}
            </a>
          ))}
        </nav>

        {/* Right-side Actions (History & Auth) */}
        <div className="header-actions">
          {/* Only show the History button if the user is logged in */}
          {user && (
            <button className="btn-history" onClick={onHistoryToggle} id="btn-toggle-history">
              History{historyCount > 0 && ` (${historyCount})`}
            </button>
          )}

          <div className="auth-section">
            {user ? (
              // Logged-in State: Show user's avatar and a sign-out button
              <div className="user-pill">
                {user.avatar ? (
                  <img src={user.avatar} alt="" className="user-avatar" referrerPolicy="no-referrer" />
                ) : (
                  <span className="user-avatar-placeholder">
                    {user.name?.[0]?.toUpperCase() || "?"}
                  </span>
                )}
                <span className="user-name">{user.name?.split(" ")[0]}</span>
                <button className="btn-signout" onClick={onSignOut} id="btn-signout" aria-label="Sign out">
                  <X size={14} />
                </button>
              </div>
            ) : (
              // Logged-out State: An empty div where Google's JavaScript will inject the login button
              <div id="google-signin-btn" />
            )}
          </div>
        </div>
      </div>
    </header>
  );
}
