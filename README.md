[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

Project 1: Navigation
-----------

Train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]
 
    
   * **Task:** Episodic
   * **Reward:** +1 for collecting a yellow banana, -1 is provided for collecting a blue banana.  
   * **State space:** 
      * **Vector Environment:** 37 dimensions that includes agent's velocity, along with ray-based perception of objects around agent's forward direction.  
      * **Visual Environment:** (84, 84, 3) where 84x84 is the image size.
   * **Action space:** 
       - **`0`** - move forward.
       - **`1`** - move backward.
       - **`2`** - turn left.
       - **`3`** - turn right.
 