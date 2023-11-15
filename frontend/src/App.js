import './App.css';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import Board from './components/Board';
import Home from './pages/Home';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';

function App() {
  return (
    <div className="App">
      <Router> 
        <Navbar />
        <Routes>
          <Route path="/" exact component={Home} />
        </Routes>
        <Board />
        <Footer />
      </Router>
    </div>
  );
}

export default App;
