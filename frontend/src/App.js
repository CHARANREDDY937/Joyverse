import React from "react";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import MainPage from "./components/MainPage";
import ChildDashboard from "./components/ChildDashboard";

const App = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<MainPage />} />
        <Route path="/child-dashboard" element={<ChildDashboard />} />
      </Routes>
    </Router>
  );
};

export default App;
