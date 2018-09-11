;; domain definition for Grid Minecraft environment

(define (domain GridMinecraft)

    (:requirements
    ;    :durative-actions
    ;    :equality
    ;    :negative-preconditions
    ;    :numeric-fluents
    ;    :object-fluents
        :typing
    )

    (:types key chest sword gate ggate)

    ; (:constants
    ; 
    ; )

    (:predicates
        (keyExists ?key - key)
        (actorWithKey)
        (chestExists ?chest - chest)
        (actorWithSword)
        (freeGateExists ?gate - gate)
        (guardedGateExists ?ggate - ggate)
        (actorReachedGate)
    )

    ; (:functions
    ; 
    ; )

    (:action grabKey
        ; grab a key
        :parameters (?key - key)
        :precondition (and
            (keyExists ?key)
            (not (actorWithKey))
        )
        :effect (and
            (actorWithKey)
            (not (keyExists ?key))
        )
    )

    (:action openChest
        ; open a chest to obtain a magic sword
        :parameters (?key - key ?chest - chest)
        :precondition (and
            (chestExists ?chest)
            (actorWithKey)
            (not (actorWithSword))
        )
        :effect (and
            (not (actorWithKey))
            (not (chestExists ?chest))
            (actorWithSword)
        )
    )

    (:action goToFreeGate
        ; go the a free gate without being guarded by a monster
        :parameters (?gate - gate)
        :precondition (and
            (freeGateExists ?gate)
            (not (actorReachedGate))
        )
        :effect (and
            (actorReachedGate)
        )
    )

    (:action goToGuardedGate
        ; go the a gate guarded by a monster
        :parameters (?ggate - ggate)
        :precondition (and
            (guardedGateExists ?ggate)
            (actorWithSword)
            (not (actorReachedGate))
        )
        :effect (and
            (actorReachedGate)
        )
    )
)
