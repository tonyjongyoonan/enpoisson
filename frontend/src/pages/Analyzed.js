import React, { useState, useEffect, useRef, Fragment } from 'react';
import Select from 'react-select';
import { useLocation } from 'react-router-dom';
import { Chessboard } from 'react-chessboard';
import { Chess } from 'chess.js';
import AnalysisMoves from './AnalysisMoves';
import Bar from '../components/Bar';
import './Analyzed.css';

const options = [
  { value: 'game', label: 'Played move'}, 
  { value: '1500', label: '1500 ELO most human move'}
]

const Analyzed = () => {
  const [fen, setFen] = useState('start');
  const chess = useRef(new Chess());
  const [index, setIndex] = useState(0);
  // const moveIndex = useRef(0);
  const [moves, setMoves] = useState([]);
  const location = useLocation();
  const pgn = location.state.pgn;
  const [selected, setSelected] = useState({ value: 'game', label: 'Played move'});
  const [arrows, setArrows] = useState([]);
  const [feedback, setFeedback] = useState("");
  const [recMove, setRecMove] = useState("");

  const getPgnMoves = (pgn) => {
    const newChess = new Chess();
    newChess.loadPgn(pgn);
    console.log(newChess.history());
    return newChess.history();
  };

  const handleChange = (option) => {
    if (option.value !== 'game') {
      setTimeout(() => {
        getEngineMove();
      }, 10)
    }
    setSelected(option);
    setArrows([]);
    setFeedback("");
  }

  const generateExplanation = () => {
    setTimeout(() => {
      getExplanation();
    }, 500)
    // setFeedback("The castle move by Black (O-O-O), though it appears to result in some slight short term losses, might set up a more advantageous strategic situation in the long run. The material count remains unchanged at 5559, indicating no pieces were lost or exchanged in this move. The move increases Black's control over the board from 37 to 46 squares, implying that Black has expanded their influence on the game. While the castle move moved the king further away from the nearest black pawn, thus potentially leading to a slightly more exposed position, the shift in control indicates a broader control of the board. However, the move leaves one of Black's pieces hanging, potentially creating a risk for the next moves. This move might signal a transition from a defensive to more offensive play for Black.")
  }

  const updateMove = (x) => {
    chess.current.reset();
    for (let i = 0; i < x; i++) {
      chess.current.move(moves[i]);
    }
    setFen(chess.current.fen());
    setIndex(x);
  }

  const makeMove = () => {
    if (index >= moves.length) {
      return;
    }

    chess.current.move(moves[index]);
    setFen(chess.current.fen());
    setRecMove("");
    setIndex(index + 1);
  }

  const getExplanation = async() => {
    try {
      const response = await fetch("http://localhost:8000/get-explanation", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          fen: chess.current.fen(),
          move: selected.value === 'game' ? moves[index] : recMove,
          is_white_move: moves.length % 2 === 1,
        })
      });
      const data = await response.json();
      setFeedback(data);
    } catch (error) {
      console.log(error);
      setFeedback("An error occurred.");
    }
  }

  const getEngineMove = async () => {
    const no_moves = chess.current.history().length;
    try {
      const response = await fetch("http://localhost:8000/get-human-move", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          fen: chess.current.fen(), 
          last_16_moves: chess.current.history().slice(Math.max(0, no_moves - 16), no_moves),
          is_white_move: moves.length % 2 !== 1 // if odd number of moves, then return false (black) since we want model to give black move
        })
      });
      const data = await response.json();
      const returned_moves = Object.keys(data);
      const probabilities = Object.values(data);
      const sumProb = probabilities.reduce((a, b) => a + b, 0);
      for (let i = 0; i < probabilities.length; i++) {
        probabilities[i] /= sumProb;
      }
      console.log(returned_moves);
      console.log(probabilities);
      const threshold = Math.random();
      console.log("threshold: " + threshold);
      let runningProb = probabilities[0];
      let selectedMove = null;
      for (let i = 0; i < returned_moves.length; i++) {
        console.log("runningProb: " + runningProb);
        if (runningProb > threshold) {
          selectedMove = returned_moves[i];
          break;
        }
        runningProb += probabilities[i + 1];
      }

      if (selectedMove) {
        setRecMove(selectedMove);
      } else {
        console.log('error: no move selected');
      }
    } catch (error) {
      console.log(error);
    }
  }

  const undoMove = () => {
    if (index <= 0) {
      return;
    }
    chess.current.undo();
    setFen(chess.current.fen());
    setRecMove("");
    setIndex(index - 1);
  }

  useEffect(() => {
    const handleKeydown = (event) => {
      if (event.key === 'ArrowRight' && index < moves.length) {
        makeMove();
      } else if (event.key === 'ArrowLeft' && index > 0) {
        undoMove();
      }
    };

    window.addEventListener('keydown', handleKeydown);
    return () => window.removeEventListener('keydown', handleKeydown);
  }, [moves, index])

  useEffect(() => {
    const temp = getPgnMoves(pgn);
    setMoves(temp); // format: ['e4', 'e5', ..., 'Nf3', 'Nc6']
  }, [pgn]);

  useEffect(() => {
    setRecMove("");
    if (selected.value !== 'game') {
      setTimeout(() => {
        getEngineMove();
      }, 10)
    }
  }, [index])

  useEffect(() => {
    if (index >= moves.length) {
      setArrows([]);
    } else if (selected.value === "game" && moves.length > 0) {
      const move_info = chess.current.move(moves[index]);
      setArrows([[move_info["from"], move_info["to"]]])
      chess.current.undo();
    } else if (recMove !== "") {
      const move_info = chess.current.move(recMove);
      setArrows([[move_info["from"], move_info["to"]]])
      chess.current.undo();
    }
  }, [selected, index, moves, recMove])

  return (
    <div className="analysis-page-layout">
      <div className="analysis-header">
        <h2>Analysis</h2>
      </div>
      <div className="analysis-page-container">
        {/* <Progress.Line showInfo={false} strokeColor={"white"} vertical={true}/> */}
        <Bar color={"black"} value={25} label={"easy"}/>
        <div className="chessboard-container">
          <Chessboard position={fen} areArrowsAllowed={false} arePiecesDraggable={false} boardWidth={560} customArrows={arrows}/>
        </div>
        <div className="analysis-right-container">
          <AnalysisMoves moves={moves} index={index} updateMove={updateMove}/>
          <div className="analysis-button-container">
            <button className="analysis-back"><img src="/left.png" onClick={undoMove} alt="left" height="50"/></button>
            <button className="analysis-move"><img src="/right.png" onClick={makeMove} alt="right" height="50"/></button>
          </div>
        </div>
        <div className="analysis-explanation-container">
          <div className="explanation-selector-container">
            <Select className="analysis-select" value={selected} onChange={handleChange} options={options} />
            { selected.value !== "game" ? 
              <p>{ chess.current.isCheckmate() ? "Checkmate!" : "Most likely move: " + recMove } </p> :
              <p>{ index < moves.length ? "Next played move: " + moves[index] : "Game is over!" }</p>
            }
            <button onClick={(e) => generateExplanation()}>Explain</button>
          </div>
          {feedback.length > 0 ? 
          <div className="explanation-container">
            <span>{feedback}</span>
          </div> 
          : <Fragment />}
        </div>
      </div>
    </div>
  );
};  
export default Analyzed;