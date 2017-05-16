#função similar ao quadprog do MatLab para minimização de funções da forma
#f(x) = (1/2)*x'*H*x + f'*x, onde H é uma matriz e f é um vetor com restrições
#da forma Ax <= b, Aeq*x = beq, lb <= x <= ub.

using JuMP, Ipopt

function quadprog(H,f; A=[],b=[],Aeq = [],beq = [],lb = [],ub = [],
  usesolver = IpoptSolver(), starter = zeros(size(H,2)))

  r,s = size(H)

  m = Model(solver = usesolver)

  if starter == zeros(size(H,2))
    @variable(m,x[1:s])
  else
    @variable(m, x[i=1:s], start = starter[i])
  end

  if !(isempty(A)) || !(isempty(b))
    @constraintref loopConstraint1[1:length(b)]
    for i = 1:length(b)
      loopConstraint1[i] = @constraint(m,dot(A[i,:],x) <= b[i])
    end
  end

  if !(isempty(Aeq)) || !(isempty(beq))
    @constraintref loopConstraint2[1:length(beq)]
    for i = 1:length(beq)
      loopConstraint2[i] = @constraint(m, dot(Aeq[i,:],x) == beq[i])
    end
  end

  if !(isempty(lb))
    @constraintref loopConstraint3[1:length(lb)]
    for i = 1:length(lb)
      loopConstraint3[i] = @constraint(m,x[i] >= lb[i])
    end
  end

  if !(isempty(ub))
    @constraintref loopConstraint3[1:length(ub)]
    for i = 1:length(ub)
      loopConstraint4[i] = @constraint(m,x[i] <= ub[i])
    end
  end

  @objective(m, Min, 0.5*dot(H*x,x) + dot(f,x))

  print(m)

  solve(m)

  return getvalue(x)
end
