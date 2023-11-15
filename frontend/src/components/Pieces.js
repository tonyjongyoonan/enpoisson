import React from 'react'
import Piece from './Piece'
import '../styles/Pieces.css'
import { useState } from 'react'
import { useRef  } from 'react'

function Pieces() {

    const createPosition = () => {
        const position = new Array(8).fill('').map(x => new Array(8).fill(''))
        for (let i = 0; i < 8; i++) {
            position[1][i] = 'wp'
            position[6][i] = 'bp'
        }

        position[0][0] = 'wr'
        position[0][7] = 'wr'
        position[7][0] = 'br'
        position[7][7] = 'br'
        position[0][1] = 'wn'
        position[0][6] = 'wn'
        position[7][1] = 'bn'
        position[7][6] = 'bn'
        position[0][2] = 'wb'
        position[0][5] = 'wb'
        position[7][2] = 'bb'
        position[7][5] = 'bb'
        position[0][3] = 'wq'
        position[0][4] = 'wk'
        position[7][3] = 'bq'
        position[7][4] = 'bk'
        return position
    }
    const ref = useRef()
    const [state, setState] = useState(createPosition())

    const onDrop = e => {
        const newPosition = new Array(8).fill('').map(x => new Array(8).fill(''))
        for (let rank = 0; rank < 8; rank++) {
            for (let file = 0; file < 8; file++) {
                newPosition[rank][file] = state[rank][file]
            }
        }

        const {top,left,width} = ref.current.getBoundingClientRect()
        const size = width / 8
        const y = Math.floor((e.clientX - left) / size) 
        const x = 7 - Math.floor((e.clientY - top) / size)

        const [p,rank,file] = e.dataTransfer.getData('text').split(',')
        newPosition[rank][file] = ''
        newPosition[x][y] = p
        setState(newPosition)

    }
    const onDragOver = e => {
        e.preventDefault()
    }

    return <div 
        className='pieces' 
        ref={ref} 
        onDrop={onDrop} 
        onDragOver={onDragOver} > 
        {state.map((r,rank) => 
            r.map((f,file) => 
                state[rank][file]
                ?   <Piece 
                        key={rank+'-'+file} 
                        rank = {rank}
                        file = {file}
                        piece = {state[rank][file]}
                    />
                :   null
            )   
        )}
    </div>
}

export default Pieces