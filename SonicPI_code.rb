





use_synth :hollow
with_fx :reverb, mix: 0.7 do
  
  
  live_loop :note1 do
    a, b, c,f = sync "/osc*/trigger/prophet"
    play choose([a,b]), attack: 6, release: 6
    sleep 1
  end
  
  live_loop :note2 do
    a, b, c,f = sync "/osc*/trigger/prophet"
    play choose([a+4,a+5]), attack: 4, release: 5
    sleep 1
  end
  
  live_loop :note3 do
    a, b, c,f = sync "/osc*/trigger/prophet"
    play choose([a+7, a+11]), attack: 5, release: 5
    sleep 1
  end
  
  live_loop :bass do
    use_synth :fm
    f, a,b,c = sync "/osc*/trigger2/prophet"
    play f-12, attack: 0, release: 0.25
    sleep 0.25
    play f-12, attack: 0, release: 0.3
    sleep 2
    play f-8
    sleep 0.75
    play f-9
    sleep 1
    play f-12, attack: 0, release: 0.6
    sleep 1
  end
  
end

