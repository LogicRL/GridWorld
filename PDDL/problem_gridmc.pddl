;; problem definition for Grid Minecraft

(define (problem ProblemGridMinecraft)

    (:domain GridMinecraft)
    
    (:objects
        key1 - key
        chest1 - chest
        sword1 - sword
        ggate1 - ggate
    )
    
    (:init
        (keyExists key1)
        (chestExists chest1)
        (guardedGateExists ggate1)
    )
    
    (:goal (and
        (actorReachedGate)
    ))
)
