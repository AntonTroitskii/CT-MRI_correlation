import math

import numpy as np
import pandas as pd
import scipy.stats as stats

import additional.add_func as funcs
import matplotlib.pyplot as plt

dic_keys = {'name', 'eq', 'func_ap', 'std_se', 'r2', 'err'}


def regression_factory(name, y, x):
    if name == 'lin':
        return LinRegression(y, x)
    elif name == 'pow':
        return PowRegression(y, x)
    elif name == 'exp':
        return ExpRegression(y, x)
    elif name == 'hyp':
        return HypRegression(y, x)
    else:
        raise Exception("Unrecognized chart style.")


class Regression:
    conf_int: funcs.ConfidenceInterval

    def __init__(self, y, x):
        self.x = x
        self.y = y

        self.name = None
        self.equation = None
        self.func_ap = None
        self.std_se = None
        self.r2 = None
        self.err = None

        self.description = None
        self.y_ap = None

        self.conf_int = None

    def get_description(self):
        desc = []
        desc.append('{} approximation'.format(self.name))
        desc.append(self.equation)
        desc.append('Стандартная ошибка регреесси Se = {:.3f}'.format(self.std_se))
        desc.append('Коэффициент детерминации R2 = {:.2f}'.format(self.r2))
        desc.append('Средняя относительная ошибка Ei = {:.2f}'.format(self.err))

        return desc

    def output_description(self):
        for i in self.get_description():
            print(i)

    def init_param_regr(self):
        self.y_ap = [self.func_ap(i) for i in self.x]
        self.std_se = funcs.std_se(self.y, self.y_ap)
        self.r2 = funcs.r2_coefficient(self.y, self.y_ap)
        self.err = funcs.mean_rel_error_ap(self.y, self.y_ap)
        self.conf_int = funcs.ConfidenceInterval(self.x, self.y, self.func_ap)

    def get_regr_info(self):
        result_regr = dict.fromkeys(dic_keys)
        result_regr['name'] = self.name
        result_regr['eq'] = self.equation
        result_regr['func_ap'] = self.func_ap
        result_regr['r2'] = self.r2
        result_regr['std_se'] = self.std_se
        result_regr['err'] = self.err

        return result_regr


class LinRegression(Regression):

    def __init__(self, y, x):
        super(LinRegression, self).__init__(y, x)
        self.pears_c = None

        self.std_a = None
        self.std_b = None

        self.name = 'lin'
        self.a, self.b = param_ab_lin_pair_regr(y, x)
        self.equation = "y = {:.3f} + {:.3}*(x)".format(self.a, self.b)
        self.pears_c = funcs.pearson_coefficient(self.y, self.x)

        def func_lin_app(i):
            return self.a + self.b * i

        self.func_ap = func_lin_app
        self.init_param_regr()
        self.std_a, self.std_b = funcs.std_param_ab_lin_regr(self.x, self.y, self.y_ap)

    def output_description(self, alpha=0.05):
        for i in self.get_description(alpha=alpha):
            print(i)

    def get_description(self, alpha=.05):
        desc = super(LinRegression, self).get_description()

        for i in self.get_desc_stud_ab(alpha=alpha):
            desc.append(i)

        for i in self.get_desc_fish_r2(alpha=alpha):
            desc.append(i)

        for i in self.get_desc_stud_pearson_rxy(alpha=alpha):
            desc.append(i)

        return desc

    def get_desc_stud_ab(self, alpha=.05, m_=1):
        t_a = np.abs(self.a / self.std_a)
        t_b = np.abs(self.b / self.std_b)

        t_crit = stats.t.ppf(1 - alpha, len(self.y) - m_ - 1)

        desc_stud = []
        desc_stud.append('')
        desc_stud.append(
            'Оценка значимости параметров уравнения регрессии с помощью t-критерия Стьюдента (p={})'.format(alpha))
        desc_stud.append('Стандартные ошибки параметров линейной регрессии:')
        desc_stud.append(
            'std_a = : {:.3f}. Стандартное отклонение коэффициента регрессии a (y = a+bx, intercept)'.format(
                self.std_a))
        desc_stud.append(
            'std_b = : {:.3f}. Стандартное отклонение коэффициента регрессии b (y = a+bx, intercept)'.format(
                self.std_b))
        desc_stud.append('t_a = {:.2f}'.format(t_a))
        desc_stud.append('t_b = {:.2f}'.format(t_b))
        desc_stud.append('t_кр = {:.2f} (для односторонней области)'.format(t_crit))

        desc_stud.append('a = {:.2f} \u00B1 {:.2f}'.format(self.a, t_crit * self.std_a))
        desc_stud.append(
            'Доверительный интервал a = [{:.2f}, {:.2f}]'.format(self.a - t_crit * self.std_a,
                                                                 self.a + t_crit * self.std_a))
        desc_stud.append('b = {:.2f} \u00B1 {:.2f}'.format(self.b, t_crit * self.std_b))
        desc_stud.append('Доверительный интервал b = [{:.2f}, {:.2f}]'.format(self.b - t_crit * self.std_b,
                                                                              self.b + t_crit * self.std_b))
        desc_stud.append(
            'Если в границы оценки параметров регресии не попадает 0, то параметры a и b ститистически значимы.')

        return desc_stud

    def get_desc_fish_r2(self, alpha=.05, m_=1):
        n_ = len(self.y)
        f_ = self.r2 / (1 - self.r2) * (n_ - 2)
        f_crit = stats.f.ppf(1 - alpha, m_, n_ - m_ - 1)
        desc = []
        desc.append('')
        desc.append(
            'Проверка значимости коээфициента детерминации. Используется F-критерий Фишера (p={}).'.format(alpha))
        print('Проверка значимости коээфициента детерминации. Используется F-критерий Фишера.')
        if f_ > f_crit:
            desc.append('f > f_кр')
            desc.append(
                'Уравнение регрессии признается значимым (нулевая гипотеза отстутсвия связи между x и y отвергается)')
        else:
            desc.append('f < f_кр')
            desc.append(
                'Отстутствует зависимость между x и y (нулевая гипотеза отстутсвия связи между x и y отвергается)')
        desc.append('R^2 = {:.2f}, f = {:.2f}, f_кр = {:.2f}'.format(self.r2, f_, f_crit))

        return desc

    def get_desc_stud_pearson_rxy(self, alpha=.05, m_=1):
        n_ = len(self.y)
        t = np.sqrt((n_ - 2) / 1 - np.power(self.pears_c, 2))
        t_crit = stats.t.ppf(1 - alpha, n_ - 2)

        desc = []
        desc.append('')
        desc.append('Проверка значимости коэффициента корреляции Пирсона (p={})'.format(alpha))
        if t > t_crit:
            desc.append('t > t_кр')
            desc.append('Коэффициент корреляции статистически значим')
        else:
            desc.append('t < t_кр')
            desc.append('Коэффициент корреляции статистически НЕ значим')
        desc.append('r_xy = {:.2f}, t = {:.2f}, t_крит = {:.2f}'.format(self.pears_c, t, t_crit))

        return desc


class PowRegression(Regression):

    def __init__(self, y, x):
        super(PowRegression, self).__init__(y, x)
        y_p = [math.log10(i) for i in y]
        x_p = [math.log10(i) for i in x]

        self.name = 'pow'
        a, b = param_ab_lin_pair_regr(y_p, x_p)
        a, b = 10 ** a, b
        self.equation = "y = {:f} * x^({:.2f})".format(a, b)

        def func_pow_ap(i):
            return a * i ** b

        self.func_ap = func_pow_ap
        self.init_param_regr()


class ExpRegression(Regression):
    def __init__(self, y, x):
        super(ExpRegression, self).__init__(y, x)

        y_exp = [math.log10(i) for i in y]
        x_exp = x
        self.name = 'exp'
        a, b = param_ab_lin_pair_regr(y_exp, x_exp)
        a, b = 10 ** a, 10 ** b
        self.equation = "y = {:f} * {:.2f}^x".format(a, b)

        def func_exp_app(i):
            return a * b ** i

        self.func_ap = func_exp_app
        self.init_param_regr()


class HypRegression(Regression):

    def __init__(self, y, x):
        super(HypRegression, self).__init__(y, x)
        y_gip = y
        x_gip = [1 / i for i in x]

        self.name = 'hyp'
        a, b = param_ab_lin_pair_regr(y_gip, x_gip)
        self.equation = "y = {:.2f} * {:.2f}/x".format(a, b)

        def func_gip_ap(i):
            return a + b / i

        self.func_ap = func_gip_ap
        self.init_param_regr()


def param_ab_lin_pair_regr(y, x):
    x_pow_2 = [i * i for i in x]
    xy = [i * j for i, j in zip(x, y)]

    y_av = np.average(y)
    x_av = np.average(x)
    xy_av = np.average(xy)
    x_pow_2_av = np.average(x_pow_2)

    b = (xy_av - x_av * y_av) / (x_pow_2_av - x_av * x_av)
    a = y_av - b * x_av

    return a, b


class Gauss_Markov:

    def __init__(self, y, y_ap, x):
        self.y = y
        self.y_ap = y_ap
        self.x = x

        self.e = y - y_ap
        self.std_e = funcs.std_se(self.y, self.y_ap)
        self.n = len(y)

    def output_condition_description(self):
        for i in self.get_desc_gm_condition():
            print(i)

    def get_desc_gm_condition(self):
        desc = []

        description_list = [self.get_desc_01(), self.get_desc_02(), self.get_desc_03(), self.get_desc_04(),
                            self.get_desc_05()]
        for description in description_list:
            for description_str in description:
                desc.append(description_str)
            desc.append('')

        return desc

    def plot_residual(self):
        plt.hist(self.e)
        plt.title('Гистограмма сотатков регрессии')
        plt.show()

        plt.plot(range(len(self.x)), self.e)
        plt.title('График остатков регрессии')
        plt.show()

    def get_desc_01(self):
        """
            Выводит описание. Оценка случайности остаточной компоненты.
        @return: описание в виде массива строк.
        """
        desc = []
        count_turn_p = funcs.count_turn_points(self.e)
        p_crit = math.trunc(2 * (self.n - 2) / 3 - 1.96 * np.sqrt((16 * self.n - 29) / 90))

        desc.append('1) Оценка. Случайность остаточной компоненты.')
        desc.append('Количество поворотных точке p = {}'.format(count_turn_p))
        desc.append('Значение критических точек p_кр = {}'.format(p_crit))

        if count_turn_p > p_crit:
            desc.append('Остатки имеют случайный характер (p > p_кр)')
        else:
            desc.append('Остатки не имеют случайный характер (p < p_кр)')

        return desc

    def get_desc_02(self, alpha=.05):
        """
            Выводит описание. Оценка математического ожидания средней величины остатков регрессии (остаточной компоненты).
        @param alpha: - параметр для построения доверительного интервала
        @return: описание в виде массива строк.
        """
        desc = []

        e_mean = np.mean(self.e)

        t = np.abs(e_mean) / self.std_e * np.sqrt(self.n)
        t_crit = stats.t.ppf(1 - alpha, self.n - 1)

        desc.append(
            '2) Оценка. M(e_ср) = 0. Равенство нулю  метематического ожидания средней величины остаточной компоненты.')
        desc.append('e_ср = {}'.format(e_mean))
        desc.append('t_кр = {}'.format(t_crit))
        desc.append('t_расч = {}'.format(t))

        if t < t_crit:
            desc.append('Оценка средней величины остатков равной нулю ПОДТВЕРЖДАЕТСЯ (t_расч < t_кр)')
        else:
            desc.append('Оценка средней величины остатков равной нулю НЕ подтверждается (t_расч > t_кр)')
        return desc

    def get_desc_03(self, alpha=.05):
        """
            Выводит описание. Оценка постоянства дисперсии остатков регрессии.
        @param alpha: - параметр для построения доверительного интервала.
        @return: описание в виде массива строк.
        """
        desc = []

        desc.append('3) Оценка. Var(e_i) = const. Постоянство дисперсии случайного члена e_i во всех наблюдениях. ')
        desc.append('')
        for i in self.goldfield_quant_tets(alpha=alpha):
            desc.append(i)
        desc.append('')
        for i in self.rank_sperman_test(alpha=alpha):
            desc.append(i)

        return desc

    def get_desc_04(self):
        """
            Выводит описание. Оценка наличия автоколрреляции остатков регрессии.
        @return: описание в виде массива строк.
        """
        desc = []

        desc.append(
            '4) Оценка. Cov(e_i, e_j)=0, i<>j. Отсутствие автокоррелямции межуд значенимяи  ошибок e_i(остатков '
            'регрессии) во всех наблюдениях.')
        desc.append('')
        for i in self.darbin_watson_test():
            desc.append(i)

        desc.append('')
        for i in self.sing_criteria():
            desc.append(i)

        desc.append('')
        for i in self.criteria_up_down_series():
            desc.append(i)

        return desc

    def get_desc_05(self):
        """
            Выводит описание. Оценка соответствия остатков регрессии нормальному закону.
        @return: описание в виде массива строк.
        """
        desc = []
        desc.append('5) Оценка соответсвия ряда остатков регрессии закону распределения N(0, Sigma^2).')
        desc.append('')
        for i in self.rs_criteria():
            desc.append(i)
        desc.append('')

        for i in self.shapiro():
            desc.append(i)
        return desc

    def goldfield_quant_tets(self, alpha=.05, m_=1):
        """
            Выводит описание. Оценка постоянства дисперсии остатков регрессии. Метод Голдфелда-Квандта.
        @param alpha: -  параметр для построения доверительного интервала.
        @param m_: - параметр - количесвто независимых переменных. Для парной регресси равен 1.
        @return: описание в виде массива строк.
        """
        n_ = math.trunc(self.n / 3)

        # сортировка массивов по возрастанию
        x_, y_ = zip(*sorted(list(zip(self.x, self.y))))

        x1 = x_[:n_]
        y1 = y_[:n_]

        x2 = x_[-n_:]
        y2 = y_[-n_:]

        lin_reg_01 = LinRegression(y1, x1)
        lin_reg_02 = LinRegression(y2, x2)

        y1_ap = lin_reg_01.y_ap
        y2_ap = lin_reg_02.y_ap

        sum_dif_1 = funcs.sum_diff_2(y1, y1_ap)
        sum_dif_2 = funcs.sum_diff_2(y2, y2_ap)

        f_crit = stats.f.ppf(1 - alpha, n_ - m_, n_ - m_)

        if sum_dif_1 > sum_dif_2:
            f_ = abs(sum_dif_1 / sum_dif_2)
        else:
            f_ = abs(sum_dif_2 / sum_dif_1)

        desc = []

        desc.append('Тест. Голдфелда-Квандта. Goldfeld-Quandt test.')
        if f_ < f_crit:
            desc.append('Наблюдается гомоскедатичность отстатков')
        else:
            desc.append('НЕ наблюдается гомоскедатичность отстатков')
        desc.append('F_расчетное = {:2}, F_критическое = {}'.format(f_, f_crit))

        return desc

    def rank_sperman_test(self, alpha=.05, m_=1):
        """
            Выводит описание. Оценка постоянства дисперсии остатков регрессии. Метод Ранговой корреляции Спирмена.
        @param alpha: -  параметр для построения доверительного интервала.
        @param m_: - параметр - количесвто независимых переменных. Для парной регресси равен 1.
        @return: описание в виде массива строк.
        """
        df = pd.DataFrame({'x': self.x, 'y': self.y})
        df['rank_x'] = df['x'].rank(method='dense')
        regr_lin = LinRegression(df.y, df.x)
        func_ap = regr_lin.func_ap
        df = df.assign(y_ap=func_ap(df.x))
        df['e'] = self.e * -1
        df['rank_e'] = df['e'].rank(method='dense')
        df = df.assign(rank_dif=df.rank_x - df.rank_e)
        df = df.assign(rank_dif_2=df.rank_dif ** 2)
        sum_dif_2 = df.rank_dif_2.sum()

        rank_coef_spir = 1 - 6 * sum_dif_2 / (self.n ** 3 - self.n)
        t_ = abs(rank_coef_spir * math.sqrt(self.n - 2) / math.sqrt(1 - rank_coef_spir ** 2))
        t_crit = stats.t.ppf(1 - alpha / 2, self.n - m_ - 1)

        desc = []

        desc.append('Тест ранговой корреляции Спирмена.')
        if t_ < t_crit:
            desc.append('t_расч < t_крит. Коффициента ранговой корреляции Спирмена статитстически НЕ значим.')
            desc.append('Гетероскедатичность остатков отсутствует.')
        else:
            desc.append('t_расч > t_крит. Коффициента ранговой корреляции Спирмена статитстически значим.')
            desc.append('Имеется гетероскедатичность остатков.')

        desc.append('t_расч = {} , t_крит = {}'.format(t_, t_crit))
        return desc

    def darbin_watson_test(self):
        """
            Выводит описание. Оценка наличия автоколрреляции остатков регрессии. Метод Дарбина-Уотсона. 
        @return: описание в виде массива строк.
        """

        def sum_e_du_f(e):
            res_sum = 0
            for i in range(1, len(e)):
                res_sum += np.power(e[i] - e[i - 1], 2)
            return res_sum

        sum_e_du = sum_e_du_f(self.e)
        sum_e = np.sum([np.power(e, 2) for e in self.e])

        d = sum_e_du / sum_e

        desc = []
        desc.append('d = {}'.format(d))
        desc.append('Посмотреть значение в таблице при n = {}'.format(self.n))

        return desc

    def sing_criteria(self):
        """
            Выводит описание. Оценка наличия автоколрреляции остатков регрессии. Критерий знаков.
        @return: описание в виде массива строк.
        """
        desc = []
        desc.append('Критерий знаков.')

        v, tau = funcs.cout_sign_series(self.e)

        v_crit = math.trunc((self.n + 2) / 2 - 1.96 * math.sqrt(self.n - 1))
        tau_crit = 1.43 * math.log(self.n - 1)

        if (v > v_crit) and (tau < tau_crit):
            desc.append('Выполнятеяс условие ((v > v_кр) и (tau < tau_кр)). Автокорреляция отстутствует.')
        else:
            desc.append('НЕ выполнятеяс условие ((v > v_кр) и (tau < tau_кр)). Автокорреляция пристутсвует.')
            desc.append('Остатики регрессии:')
            for i in self.e:
                desc.append(i)
        desc.append('v = {}, v_кр = {}, tau = {}, tau_кр = {}'.format(v, v_crit, tau, tau_crit))
        return desc

    def criteria_up_down_series(self):
        """
             Выводит описание. Оценка наличия автоколрреляции остатков регрессии. Критерий восходящих и нисходящих серий.
         @return: описание в виде массива строк.
         """

        def tau_crit_func(n):
            if n <= 26:
                return 5
            elif n <= 153:
                return 6
            elif n <= 1170:
                return 7
            else:
                return None

        e_dif = [self.e[i] - self.e[i + 1] for i in range(0, len(self.e) - 1)]

        v, tau = funcs.cout_sign_series(e_dif)

        desc = []
        desc.append('Критерий восходящих и нисходящих серий.')
        v_crit = 2 * (self.n - 2) / 3 - 1.96 * math.sqrt((16 * self.n - 29) / 90)
        tau_crit = tau_crit_func(self.n)
        if (v > v_crit) and (tau < tau_crit):
            desc.append('Выполнятеяс условие ((v > v_кр) и (tau < tau_кр)). Автокорреляция отстутствует.')
        else:
            desc.append('НЕ выполнятеяс условие ((v > v_кр) и (tau < tau_кр)). Автокорреляция пристутсвует.')
            desc.append('Рзаность остатокв регрессии:')
            for i in e_dif:
                desc.append(i)
        desc.append('v = {}, v_кр = {}, tau = {}, tau_кр = {}'.format(v, v_crit, tau, tau_crit))
        return desc

    def rs_criteria(self):
        """
            Выводит описание. Оценка соответствия остатков регрессии нормальному закону. R/S критерий.
        @return: описание в виде массива строк.
        """
        res = max(self.e) - min(self.e)
        res /= self.std_e

        desc = []
        desc.append('Критерий R/S')
        desc.append('n = {}'.format(self.n))
        desc.append('Отношение R/S = {}'.format(res))

        return desc

    def shapiro(self, alpha=.05):
        """
            Выводит описание. Оценка соответствия остатков регрессии нормальному закону. Критерий Шапиро-Уилкса.
        @param alpha: -  параметр для построения доверительного интервала.
        @return:
        """
        stat, p = stats.shapiro(self.e)

        desc = []
        desc.append('Критерий нормальности Шапиро-Уилка.')
        desc.append('p = {}'.format(p))
        desc.append('alpha = {}'.format(alpha))

        if p > alpha:
            desc.append('+ Остаток регрессии имеет нормально распределение (неудача опровержения H0, p > alpha)')
        else:
            desc.append('- Остаток регресии НЕ имеет нормальное распределение (опровергается H0), p < alpha)')

        return desc


def comparison_regr_df(y, x):
    df = pd.DataFrame()
    ser_lin = pd.Series(LinRegression(y, x).get_regr_info())
    ser_exp = pd.Series(ExpRegression(y, x).get_regr_info())
    ser_pow = pd.Series(PowRegression(y, x).get_regr_info())
    ser_gip = pd.Series(HypRegression(y, x).get_regr_info())

    df = df.append(ser_lin, ignore_index=True). \
        append(ser_exp, ignore_index=True). \
        append(ser_pow, ignore_index=True). \
        append(ser_gip, ignore_index=True)

    df = df.sort_values(by=['r2'], ascending=False)

    df = df.drop(['func_ap'], axis=1)
    df = df[['name', 'eq', 'std_se', 'r2', 'err']]

    return df
