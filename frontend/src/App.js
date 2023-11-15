import './App.css';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import Home from './pages/Home';
import Play from './pages/Play';
import About from './pages/About';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Board from './components/Board';

function App() {
  return (
    <div className="App">
      <Router> 
        <Navbar />
        <Routes>
          <Route path="/" exact component={Home} />
          <Route path="/play" exact component={Play} />
          <Route path="/about" exact component={About} />
        </Routes>
        <Board  />
        <Footer />
      </Router>
    </div>
  );
}

export default App;
