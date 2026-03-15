import java.util.Random;

public class MemorySimulation {
    // A: 舊選擇, B: 環境改變後的新較佳選擇
    private static final int ACTION_A = 0;
    private static final int ACTION_B = 1;

    private static final int TOTAL_STEPS = 300;
    private static final int CHANGE_STEP = 120;
    private static final int RUNS = 500;

    private static final int GRATITUDE_WINDOW = 30;
    private static final int STUBBORN_START = 120;
    private static final int STUBBORN_WINDOW = 60;

    private static final int ADAPT_WINDOW = 20;
    private static final double ADAPT_THRESHOLD = 0.80;

    public static void main(String[] args) {
        Result shortResult = runExperiment("短期記憶", 0.55, 0.12, 0.15, 20260311L);
        Result longResult = runExperiment("長期記憶", 0.12, 0.01, 0.08, 20260312L);

        printResult(shortResult);
        printResult(longResult);

        printComparison(shortResult, longResult);
    }

    private static Result runExperiment(String name, double alpha, double forgetRate, double epsilon, long seed) {
        Random random = new Random(seed);
        double adaptSum = 0.0;
        double gratitudeSum = 0.0;
        double stubbornSum = 0.0;

        for (int run = 0; run < RUNS; run++) {
            Agent agent = new Agent(alpha, forgetRate, epsilon, random.nextLong());
            int postLen = TOTAL_STEPS - CHANGE_STEP;
            int[] postActions = new int[postLen];

            for (int t = 0; t < TOTAL_STEPS; t++) {
                boolean changed = t >= CHANGE_STEP;
                double rewardProbA = changed ? 0.20 : 0.80;
                double rewardProbB = changed ? 0.80 : 0.20;

                int action = agent.chooseAction();
                double reward = sampleReward(random, action == ACTION_A ? rewardProbA : rewardProbB);
                agent.update(action, reward);

                if (changed) {
                    postActions[t - CHANGE_STEP] = action;
                }
            }

            int adaptTime = computeAdaptTime(postActions, ADAPT_WINDOW, ADAPT_THRESHOLD);
            double gratitude = computeChooseARatio(postActions, 0, Math.min(GRATITUDE_WINDOW, postActions.length));

            int stubbornStart = Math.min(STUBBORN_START, postActions.length);
            int stubbornEnd = Math.min(stubbornStart + STUBBORN_WINDOW, postActions.length);
            double stubborn = computeChooseARatio(postActions, stubbornStart, stubbornEnd);

            adaptSum += adaptTime;
            gratitudeSum += gratitude;
            stubbornSum += stubborn;
        }

        return new Result(
                name,
                adaptSum / RUNS,
                gratitudeSum * 100.0 / RUNS,
                stubbornSum * 100.0 / RUNS
        );
    }

    private static double sampleReward(Random random, double probability) {
        return random.nextDouble() < probability ? 1.0 : 0.0;
    }

    private static int computeAdaptTime(int[] postActions, int window, double threshold) {
        if (postActions.length == 0) {
            return 0;
        }
        int actualWindow = Math.max(1, Math.min(window, postActions.length));

        for (int start = 0; start <= postActions.length - actualWindow; start++) {
            int chooseB = 0;
            for (int i = start; i < start + actualWindow; i++) {
                if (postActions[i] == ACTION_B) {
                    chooseB++;
                }
            }
            double ratio = (double) chooseB / actualWindow;
            if (ratio >= threshold) {
                return start + actualWindow;
            }
        }
        return postActions.length;
    }

    private static double computeChooseARatio(int[] actions, int start, int end) {
        if (end <= start) {
            return 0.0;
        }
        int chooseA = 0;
        for (int i = start; i < end; i++) {
            if (actions[i] == ACTION_A) {
                chooseA++;
            }
        }
        return (double) chooseA / (end - start);
    }

    private static void printResult(Result result) {
        System.out.println("--------------------------------------------------");
        System.out.println(result.name);
        System.out.printf("適應時間（環境於 t=%d 改變後的步數）：%.2f%n", CHANGE_STEP, result.adaptTime);
        System.out.printf("感恩分數（改變後早期仍選舊恩人的比例 %%）：%.2f%%%n", result.gratitudeScore);
        System.out.printf("固執分數（改變後較晚期仍選舊恩人的比例 %%）：%.2f%%%n", result.stubbornScore);
    }

    private static void printComparison(Result shortTerm, Result longTerm) {
        System.out.println("--------------------------------------------------");
        System.out.println("結果解讀");

        if (shortTerm.adaptTime < longTerm.adaptTime) {
            System.out.println("- 短期記憶適應得更快。");
        } else {
            System.out.println("- 長期記憶適應得更快（在此設定下較不符合預期）。");
        }

        if (shortTerm.gratitudeScore < longTerm.gratitudeScore) {
            System.out.println("- 短期記憶的感恩程度較低（較可能呈現「忘恩」）。");
        } else {
            System.out.println("- 短期記憶維持感恩更久（在此設定下較不符合預期）。");
        }

        if (longTerm.stubbornScore > shortTerm.stubbornScore) {
            System.out.println("- 長期記憶在後期適應上更固執。");
        } else {
            System.out.println("- 長期記憶較不固執（在此設定下較不符合預期）。");
        }
    }

    private static class Agent {
        private final double alpha;
        private final double forgetRate;
        private final double epsilon;
        private final Random random;
        private final double[] qValues;

        Agent(double alpha, double forgetRate, double epsilon, long seed) {
            this.alpha = alpha;
            this.forgetRate = forgetRate;
            this.epsilon = epsilon;
            this.random = new Random(seed);
            this.qValues = new double[] {0.0, 0.0};
        }

        int chooseAction() {
            if (random.nextDouble() < epsilon) {
                return random.nextBoolean() ? ACTION_A : ACTION_B;
            }
            return qValues[ACTION_A] >= qValues[ACTION_B] ? ACTION_A : ACTION_B;
        }

        void update(int action, double reward) {
            // 每一步都先進行遺忘，讓兩個行為的記憶痕跡同步衰減。
            qValues[ACTION_A] *= (1.0 - forgetRate);
            qValues[ACTION_B] *= (1.0 - forgetRate);

            // 只對本次被選擇的行為做即時學習更新。
            qValues[action] += alpha * (reward - qValues[action]);
        }
    }

    private static class Result {
        final String name;
        final double adaptTime;
        final double gratitudeScore;
        final double stubbornScore;

        Result(String name, double adaptTime, double gratitudeScore, double stubbornScore) {
            this.name = name;
            this.adaptTime = adaptTime;
            this.gratitudeScore = gratitudeScore;
            this.stubbornScore = stubbornScore;
        }
    }
}
