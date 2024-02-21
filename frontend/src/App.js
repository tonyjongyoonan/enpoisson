import "./App.css";
import Navbar from "./components/Navbar";
import Home from "./pages/Home";
import Play from "./pages/Play";
import Analyze from "./pages/Analyze";
import Analyzed from "./pages/Analyzed";
import Login from "./pages/Login";
import Account from "./pages/Account";
import NotFound from "./pages/NotFound";
import Footer from "./components/Footer";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";

function App() {
  return (
    <div>
      <Router>
        <Navbar />
        <Routes>
          <Route path="/" exact element={<Home />} />
          <Route path="/play" exact element={<Play />} />
          <Route path="/analyze" exact element={<Analyze />} />
          <Route path="/analyzed" exact element={<Analyzed />} />
          <Route path="/login" exact element={<Login />} />
          <Route path="/account" exact element={<Account />} />          
          <Route path="*" element={<NotFound />} />
        </Routes>
        <Footer />
      </Router>
    </div>
  );
}

export default App;
