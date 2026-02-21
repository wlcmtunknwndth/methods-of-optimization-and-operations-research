use meval::{Context, Expr};
use nalgebra::DVector;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ParserError {
    #[error("Ошибка парсинга выражения: {0}")]
    ParseError(String),
    #[error("Ошибка вычисления: {0}")]
    EvalError(String),
    #[error("Выражение содержит недопустимые символы")]
    InvalidExpression,
}

#[derive(Clone)]
pub struct ParsedFunction {
    expr: Expr,
    num_vars: usize,
}

impl ParsedFunction {
    pub fn new(expr_str: &str, num_vars: usize) -> Result<Self, ParserError> {
        let expr: Expr = expr_str
            .parse()
            .map_err(|e: meval::Error| ParserError::ParseError(e.to_string()))?;

        let mut ctx = Context::new();
        for i in 1..=num_vars {
            ctx.var(&format!("x{}", i), 0.0);
        }

        if expr.clone().eval_with_context(ctx).is_err() {
            return Err(ParserError::InvalidExpression);
        }

        Ok(ParsedFunction { expr, num_vars })
    }

    pub fn eval(&self, point: &DVector<f64>) -> Result<f64, ParserError> {
        if point.len() != self.num_vars {
            return Err(ParserError::EvalError(
                "Неверная размерность точки".to_string(),
            ));
        }

        let mut ctx = Context::new();
        for i in 0..self.num_vars {
            ctx.var(&format!("x{}", i + 1), point[i]);
        }

        self.expr
            .clone()
            .eval_with_context(ctx)
            .map_err(|e: meval::Error| ParserError::EvalError(e.to_string()))
    }

    pub fn gradient(&self, point: &DVector<f64>, eps: f64) -> Result<DVector<f64>, ParserError> {
        let n = point.len();
        if n != self.num_vars {
            return Err(ParserError::EvalError(
                "Неверная размерность точки".to_string(),
            ));
        }

        let mut grad = DVector::zeros(n);
        let f0 = self.eval(point)?;

        for i in 0..n {
            let mut point_plus = point.clone();
            point_plus[i] += eps;
            let f_plus = self.eval(&point_plus)?;
            grad[i] = (f_plus - f0) / eps;
        }
        Ok(grad)
    }
}
