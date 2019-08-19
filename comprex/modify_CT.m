function [CT] = modify_CT(CT)
    
    for j=1:size(CT,1)
        tmp = CT{j,4};
        
        vmax = 0;
        Z = CT{j,4};
        disp(vmax);
        for k=1:size(CT{j,4})
            if Z(k) < inf
                if vmax < Z(k)
                    vmax = Z(k);
                end
            end
        end    
        for k=1:size(CT{j,4})
            if Z(k) == inf
                Z(k) = vmax;
            end
        end 
        CT{j,4} = Z;
    end
end