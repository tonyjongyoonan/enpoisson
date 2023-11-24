import './App.css';
import Board from './components/Board/Board';
import { reducer } from './reducer/reducer'
import { useReducer } from 'react'
import { initGameState } from './constants';
import AppContext from './contexts/Context'
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import Home from './pages/Home';
import Play from './pages/Play';
import About from './pages/About';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Control from './components/Control/Control';
import TakeBack from './components/Control/bits/TakeBack';
import MovesList from './components/Control/bits/MovesList';


function App() {
    const [appState, dispatch] = useReducer(reducer, initGameState)
    const providerState = {
      appState,
      dispatch
    }
    return (
      <AppContext.Provider value={providerState}>
        <div className="App">
          <Router> 
            <Navbar />
            <Routes>
              <Route path="/" exact component={Home} />
              <Route path="/play" exact component={Play} />
              <Route path="/about" exact component={About} />
            </Routes>
            <Board />
            <Control>
                <MovesList/>
                <TakeBack/>
            </Control>
            <Footer />
          </Router>
        </div>
      </AppContext.Provider>
    )
  }
  
  export default App;
  